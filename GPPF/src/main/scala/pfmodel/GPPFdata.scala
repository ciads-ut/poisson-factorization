package data
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map
import scala.io.Source

// base class for the matrix entries of B, Y, and Z
class Input() {

	/*
	row_ids = list of lists of y indices (author ids for B, word ids for Y, document ids for Z)
	row_counts = list of lists of counts

	Access as:
	inputObj.row_ids(x_id)

	or
	inputObj.row_counts(x_id)

	where x_id corresponds to author ids for B, document ids for Y, author ids for Z
	*/

	var row_ids = new ArrayBuffer[ArrayBuffer[Int]]()
	var row_counts = new ArrayBuffer[ArrayBuffer[Int]]()
}

// class for loading and accessing B, Y, and Z
class PoissonData(trFileName1: String, trFileName2: String, trZFileName: String, predFileName1: String, predFileName2: String) {

	// Main variables
	var B = new Input()	// network matrix (N x N)
	var Y = new Input()	// corpus matrix (D x V)
	var Z = new Input()	// authorship matrix (N x D)
	var Zt = new Input()	// authorship matrix (D x N)
	var mB = new Input()	// missing data for network
	var mY = new Input()	// missing data for corpus
	var mYt = new Input()	// missing data for corpus (transposed)
	var N = 0		// number of authors in network (B)
	var D = 0		// number of documents in corpus (Y)
	var V = 0		// number of words in vocabulary (Y)
	var SN = 0		// number of entries in B
	var mSN = 0		// number of missing entries in B
	var SY = 0		// number of entries in Y
	var mSY = 0		// number of missing entries in Y

	// Counters for Z matrix
	var Ndsz = new ArrayBuffer[Double]()
	var Dnsz = new ArrayBuffer[Double]()

	def construct() {

		if (trFileName1.length() > 0) {
			// Load the B matrix
			println("Loading network: " + trFileName1)
			var trLines1 = Source.fromFile(trFileName1).getLines.filter(!_.isEmpty())
			var header = (trLines1.next().split("\t").map(_.toInt)).toList
			N = header(0)
			SN = header(1)
			fill_matrix(B, trLines1, N)
		}

		if (trFileName2.length() > 0) {
			// Load the Y matrix
			println("Loading corpus: " + trFileName2)
			var trLines2 = Source.fromFile(trFileName2).getLines.filter(!_.isEmpty())
			var header = (trLines2.next().split("\t").map(_.toInt)).toList
			D = header(0)
			V = header(1)
			SY = header(2)
			fill_matrix(Y, trLines2, D)
		}

		if (predFileName1.length() > 0) {
			// Load the missing entries for B
			println("Loading network heldout links: " + predFileName1)
			var predLines1 = Source.fromFile(predFileName1).getLines
			var header = (predLines1.next().split("\t").map(_.toInt)).toList
			mSN = header(1)
			fill_matrix(mB, predLines1, N)
		}

		if (predFileName2.length() > 0) {
			// Load the missing entries for Y
			println("Loading corpus heldout links: " + predFileName2)
			var predLines2 = Source.fromFile(predFileName2).getLines
			var header = (predLines2.next().split("\t").map(_.toInt)).toList
			mSY = header(2)
			fill_matrix(mY, predLines2, D)

			// Transpose missing Y entries (more efficient for CGPPF)
			for (w <- 0 to (V-1)) {
				mYt.row_ids += new ArrayBuffer[Int]()
				mYt.row_counts += new ArrayBuffer[Int]()
			}
			var w = 0
			for (d <- 0 to (D-1)) {
				for (nz <- 0 to (mY.row_ids(d).size-1)) {
					w = mY.row_ids(d)(nz)
					mYt.row_ids(w) += d
					mYt.row_counts(w) += mY.row_counts(d)(nz)
				}
			}
		}

		if (trZFileName.length() > 0) {
			// Load the Z matrix (trzfile has no header)
			println("Loading authors: " + trZFileName)
			var trZLines = Source.fromFile(trZFileName).getLines
			fill_matrix(Z, trZLines, N)

			// Transpose Z (improves Gibbs sampling efficiency), and initialize Z counters
			for (d <- 0 to (D-1)) {
				Dnsz += 0.0
				Zt.row_ids += new ArrayBuffer[Int]()
				Zt.row_counts += new ArrayBuffer[Int]()
			}
			var d = 0
			for (n <- 0 to (N-1)) {
				Ndsz += 0.0
				for (nz <- 0 to (Z.row_ids(n).size-1)) {
					d = Z.row_ids(n)(nz)
					Zt.row_ids(d) += n
					Zt.row_counts(d) += 1
				}
			}

			// Fill Z counters
			for (d <- 0 to (D-1)) {
				for (i <- 0 to (Zt.row_ids(d).size-1)) {
					Ndsz(Zt.row_ids(d)(i).toInt) += 1.0
					Dnsz(d) += 1.0
				}
			}
		}
	}

	// fills member variables for the Input object
	def fill_matrix(mat: Input, readFile: Iterator[String], L: Integer) {

		var(n, m, value) = (0, 0, 0)

		// first create empty rows
		for (l <- 0 to (L-1)) {
			mat.row_ids += new ArrayBuffer[Int]()
			mat.row_counts += new ArrayBuffer[Int]()
		}

		// then load the data entries
		while(readFile.hasNext) {
			// load the line, remove right whitespace, split by tabs, and map to a list of integers
			var line = readFile.next().replaceAll("\\s+$", "")
			if (!line.isEmpty) {
				var line_map = (line.split("\t").map(_.toInt)).toList
				if (line_map.size == 3) {
					n = line_map(0)
					m = line_map(1)
					value = line_map(2)
					mat.row_ids(n) += m
					mat.row_counts(n) += value
				}
				else if (line_map.size == 2) {
					n = line_map(0)
					m = line_map(1)
					mat.row_ids(n) += m
					mat.row_counts(n) += 1
				}
			}
		}
	}

	// make call to start building the data object	
	construct
}

// class for poisson.config
class Config(configFileName: String) {

	// config variable mapping
	var variables = scala.collection.mutable.Map[String, String]()	

	def construct() {
		var configLines = Source.fromFile(configFileName).getLines.filter(!_.isEmpty()).filter(!_.startsWith("#")).filter(_.contains("="))
		while(configLines.hasNext) {
			var line = configLines.next()
			// remove comments
			if (line.contains("#")) {
				line = line.substring(0, line.indexOf("#"))
			}
			if (line.contains("=")) {
				// save the variable
				var line_split = (line.split("=")).toList
				var key = line_split(0)
				var value = line_split(1)
				if (value.contains("\"")) {
					// isolate quoted contents
					value = value.substring(value.indexOf("\"")+1, value.indexOf("\"", value.indexOf("\"")+1))
				}
				else {
					// trim left and right whitespace
					value = value.replaceAll("^\\s+", "")
					value = value.replaceAll("\\s+$", "")
				}
				variables += (key -> value)
			}
		}
	}
	construct
}

