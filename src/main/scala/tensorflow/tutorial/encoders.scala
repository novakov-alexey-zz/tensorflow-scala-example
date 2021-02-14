package tensorflow.tutorial

import tensorflow.tutorial.converter._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object encoders {
  def toClasses[T: ClassTag: Ordering, U: ClassTag](
      samples: Array[T]
  ): Map[T, U] =
    samples.distinct.sorted.zipWithIndex.toMap.view
      .mapValues(transformAny[Int, U])
      .toMap[T, U]

  object LabelEncoder {
    def fit[T: ClassTag: Ordering](samples: Array[T]): LabelEncoder[T] =
      LabelEncoder[T]().fit(samples)
  }

  case class LabelEncoder[T: ClassTag: Ordering](
      classes: Map[T, T] = Map.empty[T, T]
  ) {
    def fit(samples: Array[T]): LabelEncoder[T] =
      LabelEncoder(toClasses[T, T](samples))

    def transform(t: Matrix[T], col: Int): Matrix[T] =
      t.map(
        _.zipWithIndex.map { case (d, i) =>
          if (i == col) classes.getOrElse(d, d) else d
        }
      )
  }

  object OneHotEncoder {
    def fit[T: Ordering: ClassTag, U: Numeric: Ordering: ClassTag](
        samples: Array[T]
    ): OneHotEncoder[T, U] =
      OneHotEncoder[T, U]().fit(samples)
  }

  case class OneHotEncoder[
      T: Ordering: ClassTag,
      U: Numeric: Ordering: ClassTag
  ](
      classes: Map[T, U] = Map.empty[T, U],
      notFound: Int = -1
  ) {
    def fit(samples: Array[T]): OneHotEncoder[T, U] =
      OneHotEncoder[T, U](toClasses[T, U](samples))

    def transform(a: Matrix[T], col: Int): Matrix[T] = {
      lazy val numeric = implicitly[Numeric[U]]
      val data = a.map { row =>
        row.zipWithIndex
          .foldLeft(ArrayBuffer.empty[T]) { case (acc, (d, i)) =>
            if (i == col) {
              val pos = classes.get(d)
              val zero = transformAny[Int, T](0)
              val array = Array.fill[T](classes.size)(zero)
              pos match {
                case Some(p) =>
                  array(numeric.toInt(p)) = transformAny[U, T](numeric.one)
                case None =>
                  array(0) = transformAny[U, T](numeric.fromInt(notFound))
              }
              acc ++ array
            } else acc :+ d
          }
          .toArray[T]
      }
      data
    }
  }

  case class ColumnStat(mean: Double, stdDev: Double)

  case class StandardScaler[T: Numeric: ClassTag](
      stats: Array[ColumnStat] = Array.empty
  ) {
    def fit(samples: Matrix[T]): StandardScaler[T] =
      StandardScaler(transpose(samples).map(fitColumn))

    private def transpose(a: Matrix[T]) = {
      val (rows, cols) = (a.length, a.head.length)
      val transposed = Array.ofDim[T](cols, rows)

      for (i <- (0 until rows).indices) {
        for (j <- (0 until cols).indices) {
          transposed(j)(i) = a(i)(j)
        }
      }
      transposed
    }

    private def fitColumn(data: Array[T]) = {
      val nums = data.map(transformAny[T, Double])
      val mean = nums.sum / data.length
      val stdDev = math.sqrt(
        nums.map(n => math.pow(n - mean, 2)).sum / (data.length - 1)
      )
      ColumnStat(mean, stdDev)
    }

    def transform(t: Matrix[T]): Matrix[T] = {
      val (rows, cols) = (t.length, t.head.length)
      val res = Array.ofDim[T](rows, cols)

      for (i <- 0 until rows) {
        for (j <- 0 until cols) {
          val stat = stats(j)
          val n = transformAny[T, Double](t(i)(j))
          res(i)(j) = transformAny[Double, T](scale(n, stat))
        }
      }
      res
    }

    private def scale(n: Double, stat: ColumnStat): Double =
      (n - stat.mean) / stat.stdDev
  }
}
