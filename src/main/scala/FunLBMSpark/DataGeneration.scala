package FunLBMSpark

import FunCLBMSpark.Tools._
import breeze.linalg.DenseMatrix
import breeze.numerics.sin
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.math._
//import com.github.unsupervise.spark.tss.core.TSS
import breeze.stats.distributions.Gaussian

import scala.util.Random

object DataGeneration  {

  // Sin prototype
  def f1(x: List[Double], sigma: Double): List[Double] = {
    x.map(t => 0.5*sin(4*scala.math.Pi*t) + Gaussian(0,sigma).draw())
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

    x.map(t => G.pdf(t)/G.pdf(center)+ Gaussian(0,sigma).draw())
  }


  def randomDataGeneration(prototypes: DenseMatrix[(List[Double],Double)=> List[Double]],
                           sigma: Double,
                           sizeClusterRow: List[Int],
                           sizeClusterCol: List[Int],
                           shuffled: Boolean=true,
                           numPartition: Int=200)(implicit ss: SparkSession): TSS = {


    require(sizeClusterRow.length == prototypes.rows)
    require(sizeClusterCol.length == prototypes.cols)

    val K = prototypes.rows
    val L = prototypes.cols
    val length = 100
    val indices:List[Double] = (1 to length).map(_/length.toDouble).toList

    val dataPerBlock: List[List[DenseMatrix[List[Double]]]] =
      (0 until prototypes.rows).map(k => {
        (0 until prototypes.cols).map(l => {
          val ArrayTS: Array[List[Double]] = (0 until sizeClusterRow(k) * sizeClusterCol(l)).map(e =>
            prototypes(k,l)(indices, sigma)).toArray
          DenseMatrix(ArrayTS).reshape(sizeClusterRow(k), sizeClusterCol(l))
        }).toList
      }).toList

    val data: DenseMatrix[List[Double]] = (0 until L).map(l => {
      (0 until K).map(k => {
        dataPerBlock(k)(l)
      }).reduce(DenseMatrix.vertcat(_, _))
    }).reduce(DenseMatrix.horzcat(_, _))

    val dataList: List[(Int, Int, List[Double], List[Double])] = (0 until data.rows).map(i => {
      (0 until data.cols).map(j => {
        (i, j, indices, data(i,j))
      }).toList
    }).reduce(_++_)

    val dataRDD: RDD[(Int, Int, List[Double], List[Double])] = if(shuffled){ss.sparkContext.parallelize(Random.shuffle(dataList), numPartition)}
    else ss.sparkContext.parallelize(dataList, numPartition)

    FunCLBMSpark.TSSInterface.toTSS(dataRDD)
  }
}

