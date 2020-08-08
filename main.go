package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/dgraph-io/badger/v2"
	"github.com/pkg/profile"
)

// var metric = "euclidean"
var metric = "cosine"

// Vamana is the graph constructed on the dataset
type Vamana struct {
	rawDataPath string
	s           int // index of medoid
	medoidMean  []float64
	vocabCount  int
	R           int // max out-degree of graph
	dim         int // dimension of vectors
	vocabs      []string
	vectors     [][]float64
	nout        [][]int

	M         int
	K         int
	codewords [][][]float64
}

func main() {

	defer profile.Start(profile.CPUProfile, profile.ProfilePath("./profiling")).Stop()
	rand.Seed(time.Now().UnixNano())

	vocabCount := 100000
	M := 10
	K := 256

	vam := &(Vamana{
		rawDataPath: "../raw_data/glove.6B/glove.6B.300d.txt",
		vocabCount:  vocabCount,
		R:           50,
		dim:         300,
		vocabs:      make([]string, 0, vocabCount),    // list of vocabs
		vectors:     make([][]float64, 0, vocabCount), // corresponding vectors for the vocabs
		nout:        make([][]int, 0, vocabCount),     // out-neighbors for each vocab
		M:           M,
		K:           K,
		codewords:   make([][][]float64, M), // M*K*?
	})

	var wg sync.WaitGroup

	start := time.Now()

	// Construct vamana index on the raw data
	vam.IndexRawData(func() {
		wg.Add(1)
		go vam.compressVectors("../vgraph/glove.6B.300d."+metric+"/codewords.txt", "../vgraph/glove.6B.300d."+metric+"/graph2.txt", &wg)
	})
	end := time.Now()
	fmt.Println("\nLoading and indexing data took: ", end.Sub(start))
	fmt.Println()

	start = time.Now()
	// Save the constructed index to appropriate files so we only need those files to do vector search
	vam.writeIndexToDisk("../vgraph/glove.6B.300d."+metric+"/graph1.txt",
		"../vgraph/glove.6B.300d."+metric+"/db/",
		"../vgraph/glove.6B.300d."+metric+"/medoid.txt")
	end = time.Now()
	fmt.Println("\nWriting index to disk took: ", end.Sub(start))
	fmt.Println()

	// When IndexRawData is called, it first loads the raw data into memory.
	// It then calls the callback which does vector compression as a goroutine. While this goroutine
	// is in progress, IndexRawData constructs the vamana index.
	// vam.writeIndexToDisk is executed to save the index to disk. So, we need to wait for the compression which
	// was started in IndexRawData to finish before returning. Note that vam.compressVectors also saves its resulting codebook to
	// disk after compression is done
	wg.Wait()
}

// finds codewords (K different codewords) for each subspace (M of them), writes them to file, then
// compress all vectors using the codewords and write compressed version of these vectors to file too
func (vam *Vamana) compressVectors(cwPath string, compVecPath string, wg *sync.WaitGroup) {
	defer wg.Done()
	// return
	start := time.Now()
	// find codewords
	vam.pq()

	// print these codewords
	// fmt.Println()
	// fmt.Printf("%v", vam.codewords)
	// fmt.Println()

	// write the found codewords for each subspace to file
	cwFile, err := os.Create(cwPath)
	check(err)
	defer cwFile.Close()

	cwWriter := bufio.NewWriter(cwFile)
	for m := range vam.codewords {
		for k := range vam.codewords[m] {
			cw := vam.codewords[m][k]
			_, err := cwWriter.WriteString(strconv.Itoa(m) + " " + strconv.Itoa(k) + " " + fmt.Sprintf("%f", sqNorm(cw)) + " " + vecToString(cw) + "\n")
			check(err)
		}
	}
	cwWriter.Flush()

	// compress each vector with the codewords and write to disk
	compVecFile, err := os.Create(compVecPath)
	check(err)
	defer compVecFile.Close()

	compVecWriter := bufio.NewWriter(compVecFile)
	for i, vocab := range vam.vocabs {
		enc, _ := vam.encodeVec(vam.vectors[i])
		// _, err := compVecWriter.WriteString(vocab + " " + fmt.Sprintf("%f", sqNorm(vam.vectors[i])) + " " + vam.encodeVec(vam.vectors[i]) + "\n")
		_, err := compVecWriter.WriteString(vocab + " " + fmt.Sprintf("%f", sqNorm(vam.vectors[i])) + " " + enc + "\n")
		check(err)
	}
	compVecWriter.Flush()
	end := time.Now()

	fmt.Println("COMPRESSVECTORS took: ", end.Sub(start))
}

// Split vec into vam.M subspaces and encode each as the index of
// the codeword closest to it (using sq euclidean distance)
func (vam *Vamana) encodeVec(vec []float64) (string, float64) {
	if len(vec) != vam.dim {
		log.Fatal("Expected vector of dim ", vam.dim, ", got dim ", len(vec))
	}

	encoding := ""
	var sqNormAcc float64 // sq norm of the compressed version of the vector

	ds := vam.dim / vam.M

	for m := 0; m < vam.M; m++ {

		x := vec[ds*m : ds*(m+1)]
		var closest int = 0
		var closestDist float64 = euclidean(x, vam.codewords[m][0])

		for i, val := range vam.codewords[m] {
			dist := euclidean(x, val)
			if dist < closestDist {
				closest = i
				closestDist = dist
			}
		}
		encoding += (strconv.Itoa(closest) + " ") // POINT A
		// sqNormAcc += sqNorm(vam.codewords[m][closest])
	}
	return encoding[0 : len(encoding)-1], sqNormAcc // remove trailing whitespace (from POINT A above)
}

// Find codewords and store them at vam.codewords
func (vam *Vamana) pq() {

	// M is number of subspaces
	// K is number of codewords for each subspace
	// ds is dimension of each subspace

	K := vam.K
	M := vam.M
	ds := vam.dim / M

	// vam.vocabCount should be >= K
	if vam.vocabCount < K {
		log.Fatal("Number of vectors should be greater than or equal to K, the number of codewords for each subspace")
	}

	var iwg sync.WaitGroup
	iwg.Add(M)

	for i := 0; i < M; i++ {
		go func(m int) {
			defer iwg.Done()

			centroids := vam.kMeansPlusPlusInit(K, ds, m)
			iterationCount := 0

			for iterationCount <= 30 {
				// initialize vars that will be used in calculating means of
				// vectors assigned to various centroids. These means will
				// become the new centroids
				acc := make([][]float64, K)
				for i := range acc {
					acc[i] = make([]float64, ds)
				}
				count := make([]int, K) // already initialized to 0s

				// for each vector, decide to which centroid(the nearest) it should be assigned
				// and add it to the value in the acc for that centroid
				for j := range vam.vectors {
					vec := vam.vectors[j][ds*m : ds*(m+1)]
					centroid := 0
					centroidDist := euclidean(centroids[0], vec)

					for idx, val := range centroids {
						dist := euclidean(val, vec)
						if dist < centroidDist {
							// vec is closer to this centroid than the previously closest
							centroid = idx
							centroidDist = dist
						}
					}

					// we've found centroid nearest to vec
					// add vec to this centroid's accumulator
					// then increase count of the centroid
					for f := range vec {
						acc[centroid][f] = acc[centroid][f] + vec[f]
					}
					count[centroid] = count[centroid] + 1
				}

				// use acc and count to compute new centroids
				for i := 0; i < K; i++ {
					for j := 0; j < ds; j++ {
						centroids[i][j] = acc[i][j] / float64(count[i])
					}
				}

				iterationCount++
			}

			// done finding codewords for the i-th subspace
			vam.codewords[m] = centroids
		}(i)
	}

	iwg.Wait()
}

func (vam *Vamana) kMeansPlusPlusInit(K, ds, m int) [][]float64 {

	// vam.vocabCount should be >= K

	if vam.vocabCount < K {
		log.Fatal("Number of vectors should be greater than or equal to K, the number of codewords for each subspace")
	}

	// the 256 codewords of dimension ds for the subspace i
	var centroids = make([][]float64, 0, K)

	// randomly select the first centroid
	r := rand.Intn(len(vam.vectors))
	centroids = append(centroids, vam.vectors[r][ds*m:ds*(m+1)])

	// to choose the next 256 - 1 centroids, for 256-1 times,
	// for each vector,
	// 1. find which centroid it is closest to and the sq dist from this centroid
	// 2. the vector with the largest sq dist from its
	// closest centroid is selected

	// NOTE: technically, we don't have to do the above steps
	// for points which have been selected as centroids but
	// since max sq dist is been selected for, each such point
	// will be its own closest centroid and so have 0 sq dist,
	// thus it will never be selected again as a centroid.
	// As such, this approach is harmless and we gain simplicity
	// of implementation.

	for j := 1; j < K; j++ {

		centroidCand := make([]float64, ds)
		distFromClosestCentroid := float64(0.0)

		for k := 0; k < len(vam.vectors); k++ {

			// the vector being considered for centroid-ship
			vec := vam.vectors[k][ds*m : ds*(m+1)]

			// set its closest centroid to the first centroid
			closestDist := euclidean(centroids[0], vec)

			// check if it is actually closer to another centroid
			// than that stored in closest
			for _, val := range centroids {
				dist := euclidean(vec, val)
				if dist < closestDist {
					// new closest centroid found
					closestDist = dist
				}
			}

			if closestDist > distFromClosestCentroid {
				centroidCand = vec
				distFromClosestCentroid = closestDist
			}
		}

		// we have the j+1-th centroid
		centroids = append(centroids, centroidCand)
	}

	// we have all K centroids
	return centroids
}

func check(e error) {
	if e != nil {
		log.Fatal(e)
	}
}

func (vam *Vamana) writeIndexToDisk(graphPath string, dbPath string, medoidPath string) {

	var wg sync.WaitGroup
	count := vam.vocabCount

	file, err := os.Open(vam.rawDataPath)
	check(err)
	defer file.Close()

	// store graph to disk
	graphFile, err1 := os.Create(graphPath)
	check(err1)
	defer graphFile.Close()

	// write medoid to disk
	medoidFile, err3 := os.Create(medoidPath)
	check(err3)
	defer medoidFile.Close()

	medoidWriter := bufio.NewWriter(medoidFile)
	_, e := medoidWriter.WriteString(strconv.Itoa(vam.s))
	check(e)
	medoidWriter.Flush()

	// open badger to write the graph to
	dbBufferLen := 500
	dbLastIndex := -1
	db, err := badger.Open(badger.DefaultOptions(dbPath))
	check(err)
	defer db.Close()

	scanner := bufio.NewScanner(file)
	graphWriter := bufio.NewWriter(graphFile)

	for i := 0; i < count && scanner.Scan(); i++ {
		scannerText := scanner.Text()

		graphLine := "" + scannerText
		for j, val := range vam.nout[i] {
			if j != len(vam.nout[i])-1 {
				graphLine += (" " + strconv.Itoa(val))
			} else {
				graphLine += (" " + strconv.Itoa(val) + "\n")
			}
		}

		if i%dbBufferLen == 0 && i != 0 {
			//write graph for vocabs in range [dbLastIndex, current_i]
			wg.Add(1)
			go vam.writeGraphToBadger(dbLastIndex+1, i, db, &wg)
			dbLastIndex = i
		}

		_, e1 := graphWriter.WriteString(graphLine)
		check(e1)
	}

	graphWriter.Flush()

	if dbLastIndex != count-1 {
		// there are some vocabs whose graphs have to be saved
		wg.Add(1)
		go vam.writeGraphToBadger(dbLastIndex+1, count-1, db, &wg)
	}

	wg.Wait()
}

func (vam *Vamana) writeGraphToBadger(from int, to int, db *badger.DB, wg *sync.WaitGroup) {
	wb := db.NewWriteBatch()
	defer wb.Cancel()
	defer wg.Done()

	// fmt.Println("Will write graph of vocab: ", from, " through ", to, "(exclusive)")
	for i := from; i <= to; i++ {
		value := make([]byte, 8*vam.dim)
		for j, val := range vam.vectors[i] {
			binary.BigEndian.PutUint64(value[j*8:(j+1)*8], math.Float64bits(val))
		}

		value2 := make([]byte, 4*len(vam.nout[i]))
		for k, val := range vam.nout[i] {
			binary.BigEndian.PutUint32(value2[k*4:(k+1)*4], uint32(val))
		}

		finalValue := append(value, value2...)
		wb.Set([]byte(vam.vocabs[i]), finalValue)

		// fmt.Println("key: ", vam.vocabs[i], " value: ", finalValue)
		// fmt.Println()
	}

	err := wb.Flush()
	check(err)

}

// IndexRawData constructs a vamana index on the raw data
func (vam *Vamana) IndexRawData(callback func()) {

	const lConst = 10

	//1. Initialize G to random R-regular directed graph

	// fmt.Println()
	// fmt.Println("-->Loading and initializing graph...")

	vam.LoadData(vam.vocabCount, 500, memoryInsert)
	vam.RemoveSelfFromNout()

	// fmt.Println("Vectors loaded into memory. Print them out: ", vam.vectors, " number of vectors: ", len(vam.vectors))

	// call callback to start product quantization since we're done
	// loading the vectors into memory
	callback()

	//2. Let s denote medoid
	start := time.Now()
	vam.s, vam.medoidMean = vam.findMedoid()
	end := time.Now()

	fmt.Println()
	fmt.Println("Took: ", end.Sub(start), " to find medoid.")
	// fmt.Println("Medoid: ", vam.s, "\n Medoid mean: ", vam.medoidMean)
	fmt.Println()

	for _, alpha := range []float64{1, 1.2} {

		// fmt.Println()
		// fmt.Printf("Running vamana for alpha=%v", alpha)
		// fmt.Println()

		//3. Loop throgh random permutation of 1..n
		for _, i := range rand.Perm(vam.vocabCount) {

			//4. Perform greedy search with i as query
			_, _, visited := vam.GreedySearch(vam.vectors[i], 1, lConst)

			//5. Run RobustPrune on i
			vam.RobustPrune(i, Set{ms: visited}, alpha, vam.R) // updates nout of point i

			//6. For every point in the nout RobustPrune above just found for i,
			for _, j := range vam.noutFor([]int{i}).ms {

				// we want to add i to their nouts too
				noutHopeful := vam.noutFor([]int{j}).union(Set{ms: []int{i}})

				//6.1 If adding i to j's nout makes j violate max out-deg R,
				//re-run RobustPrune for j with 'v' (visited) argument = noutHopeful
				if len(noutHopeful.ms) > vam.R {
					vam.RobustPrune(j, noutHopeful, alpha, vam.R)
				} else {

					// 6.2 R won't be violated so add i to j's nout
					sort.Ints(noutHopeful.ms)
					vam.nout[j] = noutHopeful.ms
				}
			}

			// dividend := alphaIdx*vam.vocabCount + permIdx + 1
			// divisor := 2 * vam.vocabCount
			// overallPercentageComp := (float64(dividend) * 100) / float64(divisor)

			// fmt.Println()
			// fmt.Printf("\talpha=%v \t%v / %v  vectors indexed  \n\tOverall: %v percent complete. ",
			// 	alpha, permIdx+1, vam.vocabCount, overallPercentageComp)
			// fmt.Println()
		}
	}
}

func (vam *Vamana) findMedoid() (int, []float64) {

	var acc = make([]float64, vam.dim)
	for _, vec := range vam.vectors {
		for i := 0; i < vam.dim; i++ {
			acc[i] += vec[i]
		}
	}

	var mean = make([]float64, vam.dim)
	for i := 0; i < vam.dim; i++ {
		mean[i] = acc[i] / float64(vam.vocabCount)
	}

	// find which of the vectors is closest to the mean (variable acc)
	var closest int = 0
	var dist float64 = euclidean(vam.vectors[0], acc)

	for i := 1; i < len(vam.vectors); i++ {
		d := euclidean(vam.vectors[i], acc)
		if d < dist {
			// this vector is better suited to be medoid
			closest = i
			dist = d
		}
	}

	return closest, mean
}

// LoadData continually calls passed function(memoryInsert) after each |n| lines read from data source.
func (vam *Vamana) LoadData(lineCount int, interjectAfter int, callback func([]string, *Vamana)) {
	file, err := os.Open(vam.rawDataPath)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	buf := make([]string, 0)

	for i := 0; i < lineCount && scanner.Scan(); i++ {
		remainder := i % interjectAfter
		if remainder == 0 && i != 0 {
			callback(buf, vam)
			buf = make([]string, 0) //reset buf
		}
		buf = append(buf, scanner.Text())
	}

	if len(buf) > 0 {
		callback(buf, vam)
	}
}

// puts vocabs and their corresponding vectors and randomly-generated out-neighbors
// in memory, specifically vam.vocabs, vam.vectors, vam.nout
func memoryInsert(lines []string, vam *Vamana) {
	for _, line := range lines {
		parts := strings.Fields(line)

		vocab := parts[0]

		vector := make([]float64, vam.dim)
		for i := 0; i < vam.dim; i++ {
			num, err := strconv.ParseFloat(parts[i+1], 64)
			if err != nil {
				log.Fatal("Cannot parse this string as float64:", parts[i+1])
			}
			vector[i] = num
		}

		nout := make([]int, 0, vam.R)
		for len(nout) < vam.R {
			r := rand.Intn(vam.vocabCount)
			if contains(nout, r) == false {
				nout = append(nout, r)
			}
		}

		vam.vocabs = append(vam.vocabs, vocab)
		vam.vectors = append(vam.vectors, vector)
		vam.nout = append(vam.nout, nout)
	}
}

func vecToString(a []float64) string {
	acc := ""
	for _, val := range a {
		acc = acc + fmt.Sprintf("%f ", val)
	}
	// fmt.Println("vectostring input is ", a)
	// fmt.Println()
	// fmt.Println(acc)
	// fmt.Println("blah blah blah ")
	// fmt.Println(len(acc))
	return acc[0 : len(acc)-1] // remove trailing whitespace
}

func sqNorm(a []float64) float64 {
	var acc float64

	for _, val := range a {
		acc += val * val
	}
	return acc
}

// L2 implements the L2 norm of two vectors
func L2(a, b []float64) float64 {
	if metric == "cosine" {
		return cosTheta(a, b)
	}
	return euclidean(a, b)
}

func cosTheta(a, b []float64) float64 {
	var aSum float64
	var bSum float64
	var num float64

	for i := 0; i < len(a); i++ {
		aSum += (a[i] * a[i])
		bSum += (b[i] * b[i])
		num += (a[i] * b[i])
	}

	// fmt.Println("costheta is: ", num/(math.Sqrt(aSum)*math.Sqrt(bSum)))
	// fmt.Println()
	return (num / (math.Sqrt(aSum*bSum) * math.Sqrt(bSum)))
}

// var l2counter int64 = 0

// L2 implements the L2 norm of two vectors
func euclidean(a, b []float64) float64 {
	// if len(a) != len(b) {
	// 	log.Fatal("L2 distance cannot be found for vectors of different dimensions")
	// }
	// l2counter++

	var acc float64 = 0
	for i := 0; i < len(a); i++ {
		s := a[i] - b[i]
		acc += s * s
	}
	return acc
	// return math.Sqrt(acc)
}

func (vam *Vamana) d(a, b int) float64 {
	return L2(vam.vectors[a], vam.vectors[b])
}

func (vam *Vamana) retainClosest(candidates []int, xq []float64) int {

	closest := candidates[0]
	closestDist := L2(xq, vam.vectors[candidates[0]])

	for _, val := range candidates[1:] {
		dist := L2(xq, vam.vectors[val])

		var cond bool
		if metric == "euclidean" {
			cond = dist < closestDist
		} else if metric == "cosine" {
			cond = dist > closestDist
		}

		if cond {
			closest = val
			closestDist = dist
		}
	}
	return closest
}

func (vam *Vamana) retainClosestK(count int, candidates []int, xq []float64) ([]int, []float64) {
	// fmt.Println("Retaining count: ", count)
	closest := make([]int, 0, count)
	distances := make([]float64, 0, count)

	for _, cand := range candidates {
		xCand := vam.vectors[cand]
		candDist := L2(xCand, xq)

		if len(closest) < count {
			closest = append(closest, cand)
			distances = append(distances, candDist)
		} else {

			var replaceableIndex int
			foundReplaceable := false

			for idx, val := range distances {
				var cond1 bool

				if metric == "euclidean" {
					cond1 = candDist < val
				} else if metric == "cosine" {
					cond1 = candDist > val
				}

				if cond1 {
					if foundReplaceable == false {
						foundReplaceable = true
						replaceableIndex = idx
					} else {
						// A replaceable item has previously been found.
						// Replace that replaceable with this newly found replaceable
						// only if the just found replaceable is even farther from xq than
						// the existing replaceable

						var cond2 bool

						if metric == "euclidean" {
							cond2 = distances[idx] > distances[replaceableIndex]
						} else if metric == "cosine" {
							cond2 = distances[idx] < distances[replaceableIndex]
						}

						if cond2 {
							replaceableIndex = idx
						}
					}
				}
			}

			if foundReplaceable == true {
				closest[replaceableIndex] = cand
				distances[replaceableIndex] = candDist
			}
		}
	}
	return closest, distances
}

// RemoveSelfFromNout if integer x is present in nout of item x, it is removed
func (vam *Vamana) RemoveSelfFromNout() {

	for i, val := range vam.nout {

		found := false
		var atIndex int

		for ii, iVal := range val {
			if iVal == i {
				found = true
				atIndex = ii
				break
			}
		}

		if found == true {
			vam.nout[i] = append(vam.nout[i][:atIndex], vam.nout[i][atIndex+1:]...)
		}
		sort.Ints(vam.nout[i])
	}
}

//Set implements a set of integers
type Set struct {
	ms []int
}

// NewSet creates a new set whose members are the unique elements in 'a'
func NewSet(a []int) Set {

	if len(a) == 0 {
		return Set{ms: make([]int, 0)}
	}

	ms := make([]int, 0, len(a))
	for _, val := range a {
		if contains(ms, val) == false {
			ms = append(ms, val)
		}
	}
	return Set{ms: ms}
}

func contains(a []int, test int) bool {
	for _, val := range a {
		if test == val {
			return true
		}
	}
	return false
}

func (s Set) union(a Set) Set {

	newMs := make([]int, 0, len(s.ms)+len(a.ms))
	newMs = append(newMs, s.ms...)
	for _, h := range a.ms {
		if contains(newMs, h) == false {
			newMs = append(newMs, h)
		}
	}
	return Set{ms: newMs}
}

func (s Set) minus(a Set) Set {
	ls := len(s.ms)
	newMs := make([]int, 0, ls)

	if ls == 0 || len(a.ms) == 0 {
		return s
	}

	for _, val := range s.ms {
		if contains(a.ms, val) == false {
			newMs = append(newMs, val)
		}
	}
	return Set{ms: newMs}
}

func (vam *Vamana) noutFor(points []int) Set {
	if len(points) == 1 {
		return Set{ms: vam.nout[points[0]]}
	}

	nout := make([]int, 0, vam.R*len(points))
	for _, p := range points {
		for _, h := range vam.nout[p] {
			if contains(nout, h) == false {
				nout = append(nout, h)
			}
		}
	}
	return Set{ms: nout}
}

// GreedySearch implements algorithm 1 in the paper
func (vam *Vamana) GreedySearch(xq []float64, k int, lConst int) ([]int, []float64, []int) {

	if lConst < k {
		log.Fatal("lConst cannot be less than k. It should at least be equal to k.")
	}
	s := vam.s

	//1. Initialize sets l = {s} and v={}
	l := Set{ms: []int{s}}
	v := Set{ms: []int{}}

	//2. While l-v is not empty, do:
	lmv := l.minus(v)

	for len(lmv.ms) > 0 {

		//3. Find |beamFactor| element(s) in set lmv.ms closest to xq. Call it/them p*
		pStar := vam.retainClosest(lmv.ms, xq)

		//4. Perform updates:  l = l U nout(p*) and v = v U {pStar}
		pStarNout := vam.nout[pStar]
		l.ms = append(l.ms, pStarNout...)
		v.ms = append(v.ms, pStar)

		//5. if |  l| > lConst, update l to retain only lConst points that are closest to xq
		if len(l.ms) > lConst {
			retained, _ := vam.retainClosestK(lConst, l.ms, xq)
			l = Set{ms: retained}
		}
		lmv = l.minus(v)
	}

	closest, distances := vam.retainClosestK(k, l.ms, xq)
	return closest, distances, v.ms
}

// RobustPrune updates nout of point p
func (vam *Vamana) RobustPrune(p int, v Set, alpha float64, R int) {
	xp := vam.vectors[p]
	v = (Set{ms: append(v.ms, vam.nout[p]...)}).minus(Set{ms: []int{p}})
	newNout := Set{ms: make([]int, 0, vam.R)}

	for len(v.ms) > 0 {
		pStar := vam.retainClosest(v.ms, xp)
		newNout.ms = append(newNout.ms, pStar)

		if len(newNout.ms) == R {
			break
		}

		for _, pBar := range v.ms {

			var cond bool

			if metric == "euclidean" {
				cond = alpha*vam.d(pStar, pBar) <= vam.d(p, pBar)
			} else if metric == "cosine" {
				cond = alpha*vam.d(pStar, pBar) >= vam.d(p, pBar)
			}
			if cond {
				// remove pBar from v
				v = v.minus(Set{ms: []int{pBar}})
			}
		}
	}

	// Replace point p's nout with newNout
	sort.Ints(newNout.ms)
	vam.nout[p] = newNout.ms
}
