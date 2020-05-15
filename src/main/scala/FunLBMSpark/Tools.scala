package FunLBMSpark

import breeze.linalg.{DenseMatrix, max}
import breeze.stats.distributions.RandBasis

object Tools extends java.io.Serializable {
  implicit val basis: RandBasis = RandBasis.withSeed(2)

  def allEqual[T](x: List[T], y:List[T]): Boolean = {
    require(x.length == y.length)
    val listBool = x.indices.map(i => {x(i)==y(i)})

    listBool.forall(identity)
  }

  def getCondBlockPartition(rowPartition: List[List[Int]], colPartition: List[Int]): List[(Int, Int)] = {

    val blockPartitionMat = colPartition.par.map(l => {
      DenseMatrix.tabulate[(Int, Int)](rowPartition.head.length, 1) {
        (i, j) => (rowPartition(l)(i), l)
      }
    }).reduce(DenseMatrix.horzcat(_,_))

    blockPartitionMat.t.toArray.toList
  }

  def printModel(model: LatentBlockModel): Unit={
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

  def updateModel(formerModel: LatentBlockModel,
                  colClusterToUpdate: Int,
                  newModel:LatentBlockModel): LatentBlockModel = {

    require(colClusterToUpdate >= 0 & colClusterToUpdate<formerModel.proportionsCols.length,
      "Col Cluster Idx to replace should be > 0 and lesser than the column cluster number of former model")

    val newRowProportion = remove(formerModel.proportionsRows, colClusterToUpdate) ++ newModel.proportionsRows
    val newColProportion = remove(formerModel.proportionsCols, colClusterToUpdate) ++
      newModel.proportionsCols.map(c => c*formerModel.proportionsCols(colClusterToUpdate))

    val newModels = remove(formerModel.gaussians, colClusterToUpdate) ++ newModel.gaussians

    LatentBlockModel(newRowProportion, newColProportion, formerModel.loadings, newModels)
  }
}
