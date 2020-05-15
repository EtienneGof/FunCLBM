package KMeansSpark

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.numerics.{pow, sqrt}
import breeze.stats.distributions.RandBasis
import org.apache.spark.rdd.RDD

object Tools extends java.io.Serializable {
  implicit val basis: RandBasis = RandBasis.withSeed(2)

  def dist(x: Array[DenseVector[Double]], y:Array[DenseVector[Double]]):Double= {
    require(x.length == y.length)
    sqrt(x.indices.map(i => sum(pow(x(i)-y(i), 2))).sum)
  }

  def sumArrayDV(x: Array[DenseVector[Double]], y:Array[DenseVector[Double]]):Array[DenseVector[Double]] = {
    require(x.length == y.length)
    x.indices.map(i => x(i)+y(i)).toArray
  }

  def distanceToCenter[T](data: RDD[(Int, T)], center: T, distance: (T, T) => Double): DenseVector[Double] = {
    DenseVector(data.map(row => {
      distance(row._2, center)
    }).collect())
  }

}
