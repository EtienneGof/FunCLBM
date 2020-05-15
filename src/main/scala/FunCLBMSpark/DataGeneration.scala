package FunCLBMSpark

import Tools._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sin

import scala.math._
import breeze.stats.distributions.MultivariateGaussian
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import com.github.unsupervise.spark.tss
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.sql.SparkSession
import breeze.stats.distributions.Gaussian
import scala.util.Random

object DataGeneration  {

  // Sin prototype
  def f1(x: List[Double], sigma: Double): List[Double] = {
    x.map(t => 1+0.5*sin(4*scala.math.Pi*t) + Gaussian(0,sigma).draw())
  }

  // Sigmoid prototype
  def f2(x: List[Double], sigma: Double): List[Double] = {
    val center = Gaussian(0.6,0.02).draw()
    val slope = 20D
    val maxVal =  Gaussian(1,0.02).draw()
    x.map(t => maxVal/(1+exp(-slope*(t-center))) + Gaussian(0,sigma).draw())
  }

  // Rectangular prototype
  def f3(x: List[Double], sigma: Double): List[Double] = {
    val start = Gaussian(0.3,0.02).draw()
    val duration = Gaussian(0.3,0.001).draw()
    x.map({
      case t if t <`start` || t>=(`start`+`duration`) => 0D + Gaussian(0,sigma).draw()
      case _ => 1D + Gaussian(0,sigma).draw()
    })
  }

  // Morlet prototype
  def f4(x: List[Double], sigma: Double): List[Double] = {
    val center = Gaussian(0.5,0.02).draw()
    x.map(t => {
      val u = (t-center)*10
      exp(-0.5*pow(u,2))*cos(5*u) + Gaussian(0,sigma).draw()
    })
  }

  // Gaussian prototype
  def f5(x: List[Double], sigma: Double): List[Double] = {
    val center = Gaussian(0.5,0.02).draw()
    val sd = 0.1
    val G = Gaussian(center, sd)

    x.map(t => 1.5*G.pdf(t)/G.pdf(center)+ Gaussian(0,sigma).draw())
  }

  // Double Gaussian prototype
  def f6(x: List[Double], sigma: Double): List[Double] = {
    val center1 = Gaussian(0.2,0.02).draw()
    val center2 = Gaussian(0.7,0.02).draw()
    val sd = 0.1
    val G1 = Gaussian(center1, sd)
    val G2 = Gaussian(center2, sd)

    x.map(t => G1.pdf(t)/G1.pdf(center1)+ G2.pdf(t)/G2.pdf(center2)+ Gaussian(0,sigma).draw() )
  }

  // y-shifted Gaussian prototype
  def f7(x: List[Double], sigma: Double): List[Double] = {
    x.map(t => 0.3+ 0.3*sin(2*scala.math.Pi*t) + Gaussian(0,sigma).draw())
  }

  // sin by rectangular prototype
  def f8(x: List[Double], sigma: Double): List[Double] = {
    val start = Gaussian(0.3,0.02).draw()
    val duration = Gaussian(0.3,0.001).draw()
    x.map({
      case t if t <`start` || t>=(`start`+`duration`) => -1 + 0.5* sin(2*Pi*t) + Gaussian(0,sigma).draw()
      case t => 2+  0.5*sin(2*Pi*t)  + Gaussian(0,sigma).draw()
    })
  }

  def randomDataGeneration(prototypes: List[List[(List[Double],Double)=> List[Double]]],
                           sigma: Double,
                           sizeClusterRow: List[List[Int]],
                           sizeClusterCol: List[Int],
                           shuffled: Boolean=true,
                           numPartition: Int=200)(implicit ss: SparkSession): TSS = {

    require(Tools.allEqual(prototypes.map(_.length), sizeClusterRow.map(_.length)))
    require(sizeClusterCol.length == prototypes.length)
    require(sizeClusterRow.map(_.sum == sizeClusterRow.head.sum).forall(identity))

    val kVec = prototypes.map(_.length)
    val l = prototypes.length
    val length = 100
    val indices:List[Double] = (1 to length).map(_/length.toDouble).toList

    val dataPerBlock: List[List[DenseMatrix[List[Double]]]] =
      prototypes.indices.map(l => {
        prototypes(l).indices.map(k_l => {
          val ArrayTS: Array[List[Double]] = (0 until sizeClusterRow(l)(k_l) * sizeClusterCol(l)).map(e =>
            prototypes(l)(k_l)(indices, sigma)).toArray
          DenseMatrix(ArrayTS).reshape(sizeClusterRow(l)(k_l), sizeClusterCol(l))
        }).toList
      }).toList

    var data: DenseMatrix[List[Double]] = (0 until l).map(l => {
      (0 until kVec(l)).map(k_l => {
        dataPerBlock(l)(k_l)
      }).reduce(DenseMatrix.vertcat(_, _))
    }).reduce(DenseMatrix.horzcat(_, _))

    data = if(shuffled){
     val shuffledColdata: DenseMatrix[List[Double]] = DenseMatrix(
       Random.shuffle((0 until data.cols).map(j => data(::, j))).toList:_*).t
     val shuffledRowdata: DenseMatrix[List[Double]] = DenseMatrix(
       Random.shuffle((0 until data.rows).map(i => shuffledColdata(i, ::).t)).toList:_*).t
     shuffledRowdata
    } else data

    var dataList: List[(Int, Int, List[Double], List[Double])] = (0 until data.rows).map(i => {
      (0 until data.cols).map(j => {
        (i, j, indices, data(i,j))
      }).toList
    }).reduce(_++_)

    val dataRDD = ss.sparkContext.parallelize(dataList, numPartition)

    TSSInterface.toTSS(dataRDD)
  }
}

