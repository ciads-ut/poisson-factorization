package printutilities
import java.io._

object printutils {
	def printVec(vec: Array[Double], outpath: String, K: Int) {
		val fout = new PrintWriter(new File(outpath))
		for (k <- 0 to (K-1)) {
			fout.write(vec(k) + "\n")
		}
		fout.close
	}

	def printMat(mat: Array[Array[Double]], outpath: String, K: Int, L: Int) {
		val fout = new PrintWriter(new File(outpath))
		for (k <- 0 to (K-1)) {
			fout.write(mat(0)(k).toString)
			for (l <- 1 to (L-1)) {
				fout.write("\t" + mat(l)(k))
			}
			fout.write("\n")
		}
		fout.close
	}

	def printTrFile(mat: Array[Array[Integer]], outpath: String, Xlen: Int, Ylen: Int, header: String) {
		val fout = new PrintWriter(new File(outpath))
		fout.write(header + "\n")
		for (n <- 0 to (Xlen-1)) {
			for (m <- 0 to (Ylen-1)) {
				if (mat(n)(m) > 0) {
					fout.write(n + "\t" + m + "\t" + mat(n)(m) + "\n")
				}
			}
		}
		fout.close
	}
}
