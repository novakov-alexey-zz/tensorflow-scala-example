package tensorflow.tutorial

import java.io.File
import java.nio.file.Path
import scala.io.Source
import scala.reflect.ClassTag
import scala.util.Using
import tensorflow.tutorial.TextLoader._
import tensorflow.tutorial.converter._

object TextLoader {
  val defaultDelimiter: String = ","

  def apply(rows: String*): TextLoader =
    TextLoader(data =
      rows.toArray
        .map(_.split(defaultDelimiter).toArray)
        .to(Array)
    )

  def slice[T: ClassTag](
      data: Matrix[T],
      rows: Option[(Int, Int)] = None,
      cols: Option[(Int, Int)] = None
  ): Matrix[T] =
    (rows, cols) match {
      case (Some((rowsFrom, rowsTo)), Some((colsFrom, colsTo))) =>
        sliceArr(data, (rowsFrom, rowsTo)).map(a =>
          sliceArr(a, (colsFrom, colsTo))
        )
      case (None, Some((colsFrom, colsTo))) =>
        data.map(a => sliceArr(a, (colsFrom, colsTo)))
      case (Some((rowsFrom, rowsTo)), None) =>
        sliceArr(data, (rowsFrom, rowsTo))
      case _ => data
    }

  def sliceArr[T: ClassTag](
      data: Array[T],
      range: (Int, Int)
  ): Array[T] = {
    val (l, r) = range
    val from = if (l < 0) data.length + l else l
    val to = if (r < 0) data.length + r else if (r == 0) data.length else r
    data.slice(from, to)
  }

  def column[T: ClassTag](
      data: Matrix[T],
      i: Int
  ): Array[T] = {
    val to = i + 1
    slice(data, None, Some(i, to)).flatMap(_.headOption)
  }
}

case class TextLoader(
    path: Path = new File("data.csv").toPath,
    header: Boolean = true,
    delimiter: String = TextLoader.defaultDelimiter,
    data: Array[Array[String]] = Array.empty[Array[String]]
) {

  def load(): TextLoader = {
    val data = Using.resource(Source.fromFile(path.toFile)) { s =>
      val lines = s.getLines()
      (if (header && lines.nonEmpty) lines.toArray.tail else lines.toArray)
        .map(_.split(delimiter).toArray)
    }
    copy(data = data)
  }

  /** @param from column index
    * @param to column index (excluding last index).
    *           Negative column starts from the last column and adds `cols` to get target index
    * @tparam T type of the data
    * @return
    */
  def cols[T: ClassTag](from: Int, to: Int): Array[Array[T]] =
    transform[T](slice(data, None, Some((from, to))))

  def col[T: ClassTag](i: Int): Array[T] =
    transformArr[T](TextLoader.column(data, i))
}
