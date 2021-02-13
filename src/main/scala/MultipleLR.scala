import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.ClipGradientsByGlobalNorm
import org.platanios.tensorflow.api.learn.layers.Compose
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.control_flow.CondArg._

import java.nio.file.{Paths, Path}
import org.platanios.tensorflow.data.utilities.UniformSplit
import scala.reflect.ClassTag
import converter._
import encoders._

object MultipleLR extends App {
  val features = 12
  val targets = 1
  val batchSize = 100

  val input = tf.learn.Input(FLOAT32, Shape(-1, features))
  val trainInput = tf.learn.Input(FLOAT32, Shape(-1, targets))
  val layer =
    tf.learn.Linear[Float]("Layer_0/Linear", 6) >>
      tf.learn.ReLU[Float]("Layer_0/ReLU") >>
      tf.learn.Linear[Float]("Layer_1/Linear", 6) >>
      tf.learn.ReLU[Float]("Layer_1/ReLU") >>
      tf.learn.Linear[Float]("OutputLayer/Linear", 1) >>
      tf.learn.Sigmoid[Float]("OutputLayer/Sigmoid")

  val loss = tf.learn.L2Loss[Float, Float]("Loss/L2Loss") >>
    tf.learn.ScalarSummary[Float]("Loss/Summary", "Loss")
  val optimizer = tf.train.Adam()

  val model = tf.learn.Model.simpleSupervised(
    input = input,
    trainInput = trainInput,
    layer = layer,
    loss = loss,
    optimizer = optimizer,
    clipGradients = ClipGradientsByGlobalNorm(5.0f)
  )

  val accMetric = tf.metrics.MapMetric(
    (v: (Output[Float], (Output[Float], Output[Float]))) => {
      val (predicted, (_, actual)) = v
      val positives = predicted > 0.5f
      val shape = Shape(batchSize, positives.shape(1))
      val binary = tf
        .select(
          positives,
          tf.fill(shape)(1f),
          tf.fill(shape)(0f)
        )
      (binary, actual)
    },
    tf.metrics.Accuracy("Accuracy")
  )

  val summariesDir = Paths.get("temp/ann")
  val (xTrain, yTrain, xTest, yTest, dataTransformer) = loadData()
  val trainFeatures = tf.data.datasetFromTensorSlices(xTrain)
  val trainLabels = tf.data.datasetFromTensorSlices(yTrain)
  val testFeatures = tf.data.datasetFromTensorSlices(xTest)
  val testLabels = tf.data.datasetFromTensorSlices(yTest)
  val trainData =
    trainFeatures
      .zip(trainLabels)
      .repeat()
      .shuffle(1000)
      .batch(batchSize)
      .prefetch(10)
  val evalTrainData =
    trainFeatures.zip(trainLabels).batch(batchSize).prefetch(10)
  val evalTestData = testFeatures.zip(testLabels).batch(batchSize).prefetch(10)

  val estimator = tf.learn.InMemoryEstimator(
    model,
    tf.learn.Configuration(Some(summariesDir)),
    tf.learn.StopCriteria(maxSteps = Some(100000)),
    Set(
      tf.learn.LossLogger(trigger = tf.learn.StepHookTrigger(100)),
      tf.learn.Evaluator(
        log = true,
        datasets =
          Seq(("Train", () => evalTrainData), ("Test", () => evalTestData)),
        metrics = Seq(accMetric),
        trigger = tf.learn.StepHookTrigger(1000),
        name = "Evaluator"
      ),
      tf.learn.StepRateLogger(
        log = false,
        summaryDir = summariesDir,
        trigger = tf.learn.StepHookTrigger(100)
      ),
      tf.learn.SummarySaver(summariesDir, tf.learn.StepHookTrigger(100)),
      tf.learn.CheckpointSaver(summariesDir, tf.learn.StepHookTrigger(1000))
    ),
    tensorBoardConfig =
      tf.learn.TensorBoardConfig(summariesDir, reloadInterval = 1)
  )

  estimator.train(
    () => trainData,
    tf.learn.StopCriteria(maxSteps = Some(10000))
  )

  println(
    s"Train accuracy = ${accuracy(xTrain, yTrain)}"
  )
  println(
    s"Test accuracy = ${accuracy(xTest, yTest)}"
  )

  // Single test
  val example = TextLoader(
    "n/a,n/a,n/a,600,France,Male,40,3,60000,2,1,1,50000,n/a"
  ).cols[String](3, -1)
  val testExample = Tensor(dataTransformer(example).map(Tensor(_)).toSeq)
    .reshape(Shape(-1, features))
  val prediction = estimator.infer(() => testExample)
  println(s"Customer exited ? ${prediction.scalar > 0.5f}")

  def accuracy(input: Tensor[Float], labels: Tensor[Float]): Float = {
    val predictions = estimator.infer(() => input.toFloat).toArray
    val correct = predictions
      .map(v => if (v > 0.5f) 1f else 0f)
      .zip(labels.toFloat.toArray)
      .foldLeft(0f) { case (acc, (yHat, y)) => if (yHat == y) acc + 1 else acc }
    correct / predictions.length
  }

  private def createEncoders[T: Numeric: ClassTag](
      data: Matrix[String]
  ): Matrix[String] => Matrix[T] = {
    val encoder = LabelEncoder.fit[String](TextLoader.column(data, 2))
    val hotEncoder = OneHotEncoder.fit[String, T](TextLoader.column(data, 1))

    val label = t => encoder.transform(t, 2)
    val onehot = t => hotEncoder.transform(t, 1)
    val typeTransform = (t: Matrix[String]) => transform[T](t)

    label andThen onehot andThen typeTransform
  }

  def loadData() = {
    val loader = TextLoader(
      Path.of("data/Churn_Modelling.csv")
    ).load()
    val featureData = loader.cols[String](3, -1)

    val encoders = createEncoders[Float](featureData)
    val numericData = encoders(featureData)
    val scaler = StandardScaler[Float]().fit(numericData)

    val prepareData = (t: Matrix[String]) => {
      val numericData = encoders(t)
      scaler.transform(numericData)
    }

    val xMatrix = prepareData(featureData)
    val xData = xMatrix.map(a => Tensor(a.toSeq)).toSeq
    val targetData = loader.col[Float](-1)

    val x = Tensor(xData).reshape(Shape(-1, features))
    val y = Tensor(targetData.toSeq).reshape(Shape(-1, targets))

    val split = UniformSplit(x.shape(0), None)
    val (trainIndices, testIndices) = split(trainPortion = 0.8f)
    val xTrain = x.gather[Int](trainIndices, axis = 0)
    val yTrain = y.gather[Int](trainIndices, axis = 0)
    val xTest = x.gather[Int](testIndices, axis = 0)
    val yTest = y.gather[Int](testIndices, axis = 0)
    (xTrain, yTrain, xTest, yTest, prepareData)
  }
}
