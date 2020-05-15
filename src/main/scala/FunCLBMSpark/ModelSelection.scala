package FunCLBMSpark

import FunLBMSpark.LatentBlockModel
import breeze.linalg.{DenseMatrix, DenseVector, argmax, max}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import Tools._
import scala.collection.mutable.ListBuffer

object ModelSelection {


  def gridSearch(data: RDD[(Int, Array[DenseVector[Double]])],
                 EMMethod: String = "SEMGibbs",
                 rangeKVec: List[Int],
                 rangeL: List[Int],
                 nConcurrent: Int = 1,
                 nTryMaxPerConcurrent: Int = 10,
                 initMethod: String = "randomPartition",
                 updateLoadingsStrategy: Int=3,
                 verbose: Boolean = false,
                 withSecurity: Boolean = true)(implicit ss: SparkSession): Map[String,Product] = {
    println(">> Grid Search launch")
    var everyCombinations: List[List[Int]] = List.empty[List[Int]]
    for(i <- rangeKVec){
      for(j <- rangeL){
        everyCombinations = everyCombinations++Tools.generateCombinationWithReplacement(i,j)}
    }
    everyCombinations = everyCombinations.distinct

    require(max(rangeL)<=5 & max(rangeKVec) <=5, "In the grid search, the maxL and KVecUpperBound seem a bit high.." +
      "The number of tested combinations is "+everyCombinations.length+"" +
      "Maybe the greedySearch would be a better choice ?" +
      "If you are certain of what you're doing, set the argument withSecurity to false.")

    if(verbose){println("Launching "+everyCombinations.length.toString+" Combinations")}
    val everyRes:List[Map[String, Product]] = everyCombinations.map(combinationKVec => {
      if(verbose){println("Trying combination ("+ combinationKVec.mkString(",")+")")}
      val CLBM = new FunCLBMSpark.CondLatentBlock().setKVec(combinationKVec).setUpdateLoadingStrategy(updateLoadingsStrategy)
      CLBM.run(data,nConcurrent=nConcurrent,nTryMaxPerConcurrent=nTryMaxPerConcurrent,initMethod=initMethod, verbose=false)
    })
    val everyLoglikelihood: DenseVector[Double] = DenseVector(everyRes.map(c => c("LogLikelihood").asInstanceOf[List[Double]].last).toArray)
    val everyICL: DenseVector[Double] = DenseVector(everyRes.map(c => c("ICL").asInstanceOf[List[Double]].last).toArray)
    if(verbose){everyICL.toArray.indices.foreach(i => {print(everyCombinations(i),
      "Loglikelihood: "+everyLoglikelihood(i).toString,
      "ICL: "+everyICL(i).toString)
      if(i==argmax(everyICL)){println("<-")} else println()})}

    everyRes(argmax(everyICL))

  }


  def bestModelAfterFunLBMGridSearch(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                                     rangeL: List[Int],
                                     rangeK: List[Int],
                                     EMMethod: String= "SEMGibbs",
                                     initMethod: String = "randomPartition",
                                     nConcurrentPerTry: Int = 2,
                                     verbose:Boolean = false,
                                     nTryMaxPerConcurrent:Int = 7,
                                     updateLoadingsStrategy:Int=3)(implicit ss: SparkSession): Map[String,Product] = {

    val dataWithDummyRowPartition = periodogram.map(r => (r._1, r._2, List(0)))

    val bestLBM = FunLBMSpark.ModelSelection.gridSearch(periodogram,"SEMGibbs",
      rangeK, rangeL, verbose= verbose,
      nConcurrentEachTest=nConcurrentPerTry,
      nTryMaxPerConcurrent=nTryMaxPerConcurrent, updateLoadingsStrategy=updateLoadingsStrategy)

    val FunLBMModel: FunLBMSpark.LatentBlockModel = bestLBM("Model").asInstanceOf[LatentBlockModel]
    val LBMColPartition = bestLBM("ColPartition").asInstanceOf[List[Int]]
    println(LBMColPartition)

    val bestL = FunLBMModel.L

    val recomposedFunCLBMModel = CondLatentBlockModel(
      List.fill(bestL){FunLBMModel.proportionsRows},
      FunLBMModel.proportionsCols,
      (0 until bestL).map(l => {
        FunLBMModel.proportionsRows.indices.map(k => {
          FunLBMModel.loadings(k)(l)
        }).toList
      }).toList,
      (0 until bestL).map(l => {
        FunLBMModel.proportionsRows.indices.map(k => {
          FunLBMModel.gaussians(k)(l)
        }).toList
      }).toList)

    val dataWithRowPartition = recomposedFunCLBMModel.drawRowPartition(dataWithDummyRowPartition,LBMColPartition)
    var modelToUpdate = recomposedFunCLBMModel

    var t0 = System.nanoTime()
    (0 until bestL).foreach(l => {
      if(verbose) {println("Gridsearching column cluster "+l.toString)}
      val currentColumnClusterData: RDD[(Int, Array[DenseVector[Double]])] = dataWithRowPartition.map(row =>
        (row._1, row._2.zipWithIndex.filter(s => LBMColPartition(s._2) == l).map(_._1)))
      val componentRes = ModelSelection.gridSearch(currentColumnClusterData, EMMethod, rangeK, List(1),
        nTryMaxPerConcurrent = 15,nConcurrent=3,   initMethod = initMethod, updateLoadingsStrategy=updateLoadingsStrategy,verbose=true)
      val componentModel = componentRes("Model").asInstanceOf[CondLatentBlockModel]
      // colClusterToUpdate is 0 because the last updated component has been put at the end of the component list.
      modelToUpdate = Tools.updateModel(modelToUpdate, 0, componentModel)
      t0 = printTime(t0, "Updating component "+l.toString)
    })

    println(LBMColPartition)
    println()
    val finalLatentBlock = new FunCLBMSpark.CondLatentBlock()
      .setKVec(modelToUpdate.KVec)
      .setInitialModel(modelToUpdate)
      .setInitialColPartition(LBMColPartition)
      .setMaxBurninIterations(7).setMaxIterations(7).setUpdateLoadingStrategy(3)
    t0 = printTime(t0, "finalLatentBlock ")

    finalLatentBlock.run(periodogram, nConcurrent = 1, nTryMaxPerConcurrent = 10, initMethod = "", verbose=verbose)

  }


  def bestModelAfterFunLBMGreedySearch(periodogram: RDD[(Int, Array[DenseVector[Double]])],
                                       maxL: Int,
                                       maxK: Int,
                                       EMMethod: String= "SEMGibbs",
                                       initMethod: String = "randomPartition",
                                       nConcurrentPerTry: Int = 2,
                                       verbose:Boolean = false,
                                       nTryMaxPerConcurrent:Int = 7)(implicit ss: SparkSession): Map[String,Product] = {

    val dataWithDummyRowPartition = periodogram.map(r => (r._1, r._2, List(0)))

    val bestLBM = FunLBMSpark.ModelSelection.greedySearch(periodogram,"SEMGibbs",
      maxL, maxK, verbose= verbose,
      nConcurrentEachTest=nConcurrentPerTry,
      nTryMaxPerConcurrent=nTryMaxPerConcurrent)

    val FunLBMModel: FunLBMSpark.LatentBlockModel = bestLBM("Model").asInstanceOf[LatentBlockModel]
    val LBMColPartition = bestLBM("ColPartition").asInstanceOf[List[Int]]
    println(LBMColPartition)

    val bestL = FunLBMModel.L

    val recomposedFunCLBMModel = CondLatentBlockModel(
      List.fill(bestL){FunLBMModel.proportionsRows},
      FunLBMModel.proportionsCols,
      (0 until bestL).map(l => {
        FunLBMModel.proportionsRows.indices.map(k => {
          FunLBMModel.loadings(k)(l)
        }).toList
      }).toList,
      (0 until bestL).map(l => {
        FunLBMModel.proportionsRows.indices.map(k => {
          FunLBMModel.gaussians(k)(l)
        }).toList
      }).toList)

    val dataWithRowPartition = recomposedFunCLBMModel.drawRowPartition(dataWithDummyRowPartition,LBMColPartition)
    var modelToUpdate = recomposedFunCLBMModel

    var t0 = System.nanoTime()
    (0 until bestL).foreach(l => {
      if(verbose) {println("Gridsearching column cluster "+l.toString)}
      val currentColumnClusterData: RDD[(Int, Array[DenseVector[Double]])] = dataWithRowPartition.map(row =>
        (row._1, row._2.zipWithIndex.filter(s => LBMColPartition(s._2) == l).map(_._1)))
      val componentRes = ModelSelection.gridSearch(currentColumnClusterData, EMMethod, (1 to maxK).toList, List(1),
        nTryMaxPerConcurrent = 15,nConcurrent=3,   initMethod = initMethod, verbose=true)
      val componentModel = componentRes("Model").asInstanceOf[CondLatentBlockModel]
      // colClusterToUpdate is 0 because the last updated component has been put at the end of the component list.
      modelToUpdate = Tools.updateModel(modelToUpdate, 0, componentModel)
      t0 = printTime(t0, "Updating component "+l.toString)
    })

    println(LBMColPartition)
    println()
    val finalLatentBlock = new FunCLBMSpark.CondLatentBlock()
      .setKVec(modelToUpdate.KVec)
      .setInitialModel(modelToUpdate)
      .setInitialColPartition(LBMColPartition)
      .setMaxBurninIterations(7).setMaxIterations(7).setUpdateLoadingStrategy(3)
    t0 = printTime(t0, "finalLatentBlock ")

    finalLatentBlock.run(periodogram, nConcurrent = 1, nTryMaxPerConcurrent = 15, initMethod = "", verbose=verbose)

  }


  def greedySearch(data: RDD[(Int, Array[DenseVector[Double]])],
                   EMMethod: String= "SEMGibbs",
                   rangeKVec: List[Int],
                   maxL: Int,
                   nConcurrent:Int=2,
                   nTryMaxPerConcurrent:Int=5,
                   initMethod:String="randomPartition",
                   verbose: Boolean = false)(implicit ss: SparkSession): Map[String,Product] = {

    val p:Int = data.take(1).head._2.length
    val n:Int = data.count().toInt

    val precRes = gridSearch(data, EMMethod, List(1), List(1), nConcurrent=nConcurrent,
      nTryMaxPerConcurrent=nTryMaxPerConcurrent, initMethod=initMethod, verbose=verbose)
    var precModel: CondLatentBlockModel = precRes("Model").asInstanceOf[CondLatentBlockModel]
    var precData = data.map(r => (r._1, r._2, List(0,0)))
    var precColPartition: List[Int] = precRes("ColPartition").asInstanceOf[List[Int]]
    var precLogLikelihood: Double = Double.NegativeInfinity
    var precIcl: Double = Double.NegativeInfinity
    var newModel = precModel
    var newData = data.map(r => (r._1, r._2, List(0,0)))
    var newColPartition: List[Int] = precRes("ColPartition").asInstanceOf[List[Int]]
    var newLogLikelihood: Double = Double.NegativeInfinity
    var newIcl: Double = Double.NegativeInfinity

    var completeLogLikelihoodList: ListBuffer[Double] = new ListBuffer[Double]():+ precModel.completelogLikelihood(precData,precColPartition)
    var iclList: ListBuffer[Double] = new ListBuffer[Double]():+ precModel.ICL(completeLogLikelihoodList.last,n,p,true)
    var candidateList = List.empty[(CondLatentBlockModel, List[Int], Double)]

    var k=1

    do {
      precModel = newModel
      precData = newData
      precColPartition = newColPartition
      precLogLikelihood = newLogLikelihood
      precIcl = newIcl

      val componentToUpdate = max(precModel.KVec.length -2,0) until precModel.KVec.length
      componentToUpdate.foreach(l => {
        println("Updating component "+l.toString)
        val currentColumnClusterData: RDD[(Int, Array[DenseVector[Double]])] = precData.map(row =>
          (row._1, row._2.zipWithIndex.filter(s => precColPartition(s._2) == l).map(_._1)))
        val candidateRes = gridSearch(currentColumnClusterData, EMMethod, rangeKVec, List(2), nConcurrent=5,
          nTryMaxPerConcurrent=10, initMethod=initMethod, verbose=verbose)
        val candidateModel = candidateRes("Model").asInstanceOf[CondLatentBlockModel]
        val candidateColPartition = candidateRes("ColPartition").asInstanceOf[List[Int]]
        val candidateICL = candidateRes("ICL").asInstanceOf[List[Double]].last
        candidateList = Tools.insert(candidateList,candidateList.length,
          (candidateModel, candidateColPartition, candidateICL))
      })

      println("CandidateList is filled, corresponding col partitions are: ")
      candidateList.map(_._2).foreach(println)

      println("Now Testing every global combination")
      val globalModelCandidates = candidateList.indices.map(l => {
        println("Combination "+l.toString)
        if(candidateList(l)._3.isNegInfinity){
          (precModel, precData, precColPartition, Double.NegativeInfinity, Double.NegativeInfinity)
        } else {
          val globalModel = Tools.updateModel(precModel, l, candidateList(l)._1)
          println("before :" + precColPartition.toArray.mkString(", "))
          val globalColPartition = Tools.mergeColPartition(precColPartition, l, candidateList(l)._2)
          println("after :"+ globalColPartition.toArray.mkString(", "))
          val globalDataWithRowPartition = globalModel.drawRowPartition(precData,globalColPartition)

          val precRowMembershipPerRow: List[List[Int]] = globalDataWithRowPartition.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList
          val precRowPartition: List[List[Int]] = precModel.KVec.indices.map(l =>
            precRowMembershipPerRow.map(rowMembership => rowMembership(l))).toList
          precRowPartition.foreach(println)

          println("Row Partition - before")
          precRowPartition.foreach(println)
          println("Row Partition - after")
          val rowMembershipPerRow: List[List[Int]] = globalDataWithRowPartition.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList
          val rowPartition: List[List[Int]] = precModel.KVec.indices.map(l =>
            rowMembershipPerRow.map(rowMembership => rowMembership(l))).toList
          rowPartition.foreach(println)

          val logLikelihood = globalModel.completelogLikelihood(globalDataWithRowPartition,colPartition=globalColPartition)
          val icl = globalModel.ICL(logLikelihood,n,p,true)
          println(globalModel.KVec, icl)
          (globalModel, globalDataWithRowPartition, globalColPartition, logLikelihood, icl)

        }
      })

      val whichBestICL: Int = argmax(DenseVector(globalModelCandidates.map(_._5).toArray))

      newModel = globalModelCandidates(whichBestICL)._1
      newData = globalModelCandidates(whichBestICL)._2
      newColPartition = globalModelCandidates(whichBestICL)._3
      newLogLikelihood = globalModelCandidates(whichBestICL)._4
      newIcl = globalModelCandidates(whichBestICL)._5

      candidateList = Tools.remove(candidateList, whichBestICL)
      completeLogLikelihoodList = completeLogLikelihoodList+= newLogLikelihood
      iclList = iclList += newIcl

      println("New Column Partition after model update :"+ newColPartition.toArray.mkString(", "))
      println("New number of Row Cluster:" + newModel.KVec.toArray.mkString(", "))
      println("New Global ICL: "+ newIcl.toString)
      k+=1

      println(newIcl, precIcl, newModel.KVec.length)
      println(newModel.KVec.length < maxL)
      println(newIcl > precIcl)
      println(newModel.KVec.length < maxL && newIcl > precIcl)
    } while (newModel.KVec.length < maxL && newIcl > precIcl)

    val rowMembershipPerRow = precData.map(row => (row._1, row._3)).collect().sortBy(_._1).map(_._2).toList
    val rowPartition = precModel.KVec.indices.map(l =>
      rowMembershipPerRow.map(rowMembership => rowMembership(l))).toList

    Map("Model" -> precModel,
      "RowPartition" -> rowPartition,
      "ColPartition" -> precColPartition,
      "LogLikelihood" -> completeLogLikelihoodList.toList.drop(1),
      "ICL" -> iclList.toList.drop(1))
  }

}
