package FunCLBMSpark

import breeze.linalg.{DenseMatrix, DenseVector, max, sum}
import breeze.numerics.{exp, log}
import breeze.stats.distributions.RandBasis
import com.github.unsupervise.spark.tss.core.TSS
import org.apache.spark.ml.linalg.{Matrices, Matrix, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.col

import scala.io.Source
import scala.util.{Success, Try}

object Tools extends java.io.Serializable {
  implicit val basis: RandBasis = RandBasis.withSeed(2)

  def allEqual[T](x: List[T], y:List[T]): Boolean = {
    require(x.length == y.length)
    val listBool = x.indices.map(i => {x(i)==y(i)})

    listBool.forall(identity)
  }

  def logSumExp(X: List[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def logSumExp(X: DenseVector[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def sample(probabilities: List[Double]): Int = {
    val dist = probabilities.indices zip probabilities
    val threshold = scala.util.Random.nextDouble
    val iterator = dist.iterator
    var accumulator = 0.0
    while (iterator.hasNext) {
      val (cluster, clusterProb) = iterator.next
      accumulator += clusterProb
      if (accumulator >= threshold)
        return cluster
    }
    sys.error("Error")
  }

  def variance(X: DenseVector[Double]): Double = {
    covariance(X,X)
  }

  def covariance(X: DenseVector[Double],Y: DenseVector[Double]): Double = {
    sum( (X- mean(X)) *:* (Y- mean(Y)) ) / (Y.length-1)
  }

  def covarianceSpark(X: RDD[((Int, Int), Vector)],
                      modes: Map[(Int, Int), DenseVector[Double]],
                      count: Map[(Int, Int), Int]): Map[(Int, Int),  DenseMatrix[Double]] = {

    val XCentered : RDD[((Int, Int), DenseVector[Double])] = X.map(d => (d._1, DenseVector(d._2.toArray) - DenseVector(modes(d._1).toArray)))

    val internProduct = XCentered.map(row => (row._1, row._2 * row._2.t))
    val internProductSumRDD: RDD[((Int,Int), DenseMatrix[Double])] = internProduct.reduceByKey(_+_)
    val interProductSumList: List[((Int,Int),  DenseMatrix[Double])] = internProductSumRDD.collect().toList

    interProductSumList.map(c => (c._1,c._2/(count(c._1)-1).toDouble)).toMap

  }

  def covariance(X: List[DenseVector[Double]], mode: DenseVector[Double], constraint: String = "none"): DenseMatrix[Double] = {

    require(List("none","independant").contains(constraint))
    require(mode.length==X.head.length)
    val XMat: DenseMatrix[Double] = DenseMatrix(X.toArray:_*)
    val p = XMat.cols
    constraint match {
      case "independant" => DenseMatrix.tabulate[Double](p,p){(i, j) => if(i == j) covariance(XMat(::,i),XMat(::,i)) else 0D}
      case _ => {
        val modeMat: DenseMatrix[Double] = DenseMatrix.ones[Double](X.length,1) * mode.t
        val XMatCentered: DenseMatrix[Double] = XMat - modeMat
        XMatCentered.t * XMatCentered
      }/ (X.length.toDouble - 1)
    }
  }

  def weightedCovariance(X: DenseVector[Double], Y: DenseVector[Double], weights: DenseVector[Double]): Double = {
    sum( weights *:* (X- mean(X)) *:* (Y- mean(Y)) ) / sum(weights)
  }

  def weightedCovariance (X: List[DenseVector[Double]],
                          weights: DenseVector[Double],
                          mode: DenseVector[Double],
                          constraint: String = "none"): DenseMatrix[Double] = {
    require(List("none","independant").contains(constraint))
    require(mode.length==X.head.length)
    require(weights.length==X.length)

    val XMat: DenseMatrix[Double] = DenseMatrix(X.toArray:_*)
    //    val p = XMat.cols
    val q = mode.length
    constraint match {
      case "independant" => DenseMatrix.tabulate[Double](q,q){(i, j) => if(i == j) weightedCovariance(XMat(::,i),XMat(::,i), weights) else 0D}
      case _ => {
        //        val XMatCentered = XMat(*, ::).map(x => x-mode)
        val modeMat: DenseMatrix[Double] = DenseMatrix.ones[Double](X.length,1) * mode.t
        val XMatCentered: DenseMatrix[Double] = XMat - modeMat
        val res = DenseMatrix((0 until XMatCentered.rows).map(i => {
          weights(i)*XMatCentered(i,::).t * XMatCentered(i,::)
        }).reduce(_+_).toArray:_*)/ sum(weights)

        res.reshape(q,q)

      }
    }
  }

  def mean(X: DenseVector[Double]): Double = {
    sum(X)/X.length
  }

  def mean(X: List[DenseVector[Double]]): DenseVector[Double] = {
    require(X.nonEmpty)
    X.reduce(_+_) / X.length.toDouble
  }

  def weightedMean(X: List[DenseVector[Double]], weights: DenseVector[Double]): DenseVector[Double] = {

    //    println("WeightedMean begins")
    //    println(X.map(_(0)))
    //    println(weights)
    require(X.length == weights.length)
    //    println("> p(zik) x p(wik)")
    //    println(weights)
    val res = X.indices.map(i => weights(i) * X(i)).reduce(_+_) / sum(weights)
    //    println("> mode")
    //    println(res)
    res
  }

  def getCondBlockPartition(rowPartition: List[List[Int]], colPartition: List[Int]): List[(Int, Int)] = {

    val blockPartitionMat = colPartition.par.map(l => {
      DenseMatrix.tabulate[(Int, Int)](rowPartition.head.length, 1) {
        (i, j) => (rowPartition(l)(i), l)
      }
    }).reduce(DenseMatrix.horzcat(_,_))

    blockPartitionMat.t.toArray.toList
  }

  def printModel(model: CondLatentBlockModel): Unit={
    println("> Row proportions:")
    model.proportionsRows.foreach(println)
    println("> Column proportions:")
    println(model.proportionsCols)
    println("> Components Parameters")
    model.gaussians.foreach(m => m.foreach(s => println(s.mean, s.cov)))
  }

  def generateCombinationWithReplacement(maxK: Int, L: Int): List[List[Int]] ={
    List.fill(L)((1 to maxK).toList).flatten.combinations(L).toList
  }

  def mergeColPartition(formerColPartition: List[Int],
                        colToUpdate: Int,
                        newColPartition: List[Int]): List[Int]={


    var newGlobalColPartition = formerColPartition
    val otherThanlMap:Map[Int,Int] = formerColPartition.filter(_!=colToUpdate).distinct.sorted.zipWithIndex.toMap
    val L = max(formerColPartition)

    var iterNewColPartition = 0
    for( j <- newGlobalColPartition.indices){
      if(formerColPartition(j)==colToUpdate){
        newGlobalColPartition = newGlobalColPartition.updated(j,newColPartition(iterNewColPartition)+L)
        iterNewColPartition +=1
      } else {
        newGlobalColPartition = newGlobalColPartition.updated(j,otherThanlMap(formerColPartition(j)))
      }
    }
    newGlobalColPartition
  }

  def remove[T](list: List[T], idx: Int):List[T] = list.patch(idx, Nil, 1)

  def insert[T](list: List[T], i: Int, values: T*) = {
    val (front, back) = list.splitAt(i)
    front ++ values ++ back
  }

  def round(x: Double, digits:Int =0): Double={
    val factor: Double = Math.pow(10,digits)
    Math.round(x*factor)/factor
  }

  def updateModel(formerModel: CondLatentBlockModel,
                  colClusterToUpdate: Int,
                  newModel:CondLatentBlockModel): CondLatentBlockModel = {

    require(colClusterToUpdate >= 0 & colClusterToUpdate<formerModel.proportionsCols.length,
      "Col Cluster Idx to replace should be > 0 and lesser than the column cluster number of former model")

    val newRowProportion = remove(formerModel.proportionsRows, colClusterToUpdate) ++ newModel.proportionsRows
    val newColProportion = remove(formerModel.proportionsCols, colClusterToUpdate) ++
      newModel.proportionsCols.map(c => c*formerModel.proportionsCols(colClusterToUpdate))
    val newLoadings = remove(formerModel.loadings, colClusterToUpdate) ++ newModel.loadings
    val newModels = remove(formerModel.gaussians, colClusterToUpdate) ++ newModel.gaussians

    CondLatentBlockModel(newRowProportion, newColProportion, newLoadings, newModels)
  }

  def isInteger(x: String) = {
    val y = Try(x.toInt)
    y match {
      case Success(x) => true
      case _ => false
    }
  }

  def inverseIndexedList(data: List[(Int, Array[DenseVector[Double]])]) = {
    val p = data.take(1).head._2.length
    (0 until p).map(j => {
      (j, data.map(row => row._2(j)).toArray)
    }).toList
  }

  def joinRowPartitionToData(data: RDD[(Int, Array[DenseVector[Double]])],
                             rowPartition: List[List[Int]],
                             n:Int)(implicit ss: SparkSession) = {

    val rowPartitionPerRow: List[(Int, List[Int])] = (0 until n).map(i =>
      (i, rowPartition.indices.map(l => rowPartition(l)(i)).toList)
    ).toList

    data.join(ss.sparkContext.parallelize(rowPartitionPerRow, 30)).
      map(r => {
      (r._1, r._2._1, r._2._2)
    })
  }

  def getBlockPartition(rowPartition: List[List[Int]], colPartition: List[Int]): Array[Int] = {

    val n = rowPartition.head.length
    val p = colPartition.length
    val blockBiPartition: List[(Int, Int)] = DenseMatrix.tabulate[(Int, Int)](n,p)(
      (i, j) => (rowPartition(colPartition(j))(i), colPartition(j))
    ).toArray.toList
    val mapBlockBiIndexToBlockNum = blockBiPartition.distinct.zipWithIndex.toMap
    blockBiPartition.map(mapBlockBiIndexToBlockNum(_)).toArray
  }

  def getEntireRowPartition(rowPartition: List[List[Int]]): Array[Int] = {
    val n = rowPartition.head.length
    val L = rowPartition.length
    val rowMultiPartition: List[List[Int]] = (0 until n).map(i => (0 until L).map(l => rowPartition(l)(i)).toList).toList

    val mapMultiPartitionToRowCluster = rowMultiPartition.distinct.zipWithIndex.toMap
    rowMultiPartition.map(mapMultiPartitionToRowCluster(_)).toArray
  }

  def printTime(t0:Long, stepName: String, verbose:Boolean = true): Long={
    if(verbose){
      println(stepName.concat(" step duration: ").concat(((System.nanoTime - t0)/1e9D ).toString))
    }
    System.nanoTime
  }

  implicit class Crossable[X](xs: Traversable[X]) {
    def cross[Y](ys: Traversable[Y]) = for { x <- xs; y <- ys } yield (x, y)
  }

  def matrixToDenseMatrix(A: Matrix): DenseMatrix[Double] = {
    val p = A.numCols
    val n = A.numRows
    DenseMatrix(A.toArray).reshape(n,p)
  }

  def denseMatrixToMatrix(A: DenseMatrix[Double]): Matrix = {
    Matrices.dense(A.rows, A.cols, A.toArray)
  }
}
