import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}


object FirstSparkApplication extends App {

  predict("data-students.json")

  /**
   * From the model, it creates a csv with all the predictions
   *
   * @param testDataName : data on which prediction is made
   */
  def predict(testDataName: String): Unit = {

    //Only for windows user I think
    System.setProperty("hadoop.home.dir", "C:\\winutils")

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("FirstSparkApplication")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // We retrieve the model
    val model = getModel(spark, "data-students.json")

    // Split the data into training and test sets (30% held out for testing)
    val testData = prepareData(spark, testDataName).randomSplit(Array(0.7, 0.3))(1)

    // Make predictions.
    val predictions = model.transform(testData).select("predictedLabel", "label", "bidFloor", "mediaIndex", "appOrSiteIndex", "area",
      "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
      "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
      "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

    saveInCsv(predictions)

    spark.stop
  }

  /**
   * Save the dataframe in a folder named "predictions_[day]_[time].csv"
   *
   * @param predictions : dataframe to save
   */
  private def saveInCsv(predictions: DataFrame): Unit = {
    val dir = new File("result").mkdir
    val date = LocalDateTime.now()
    val dateFormat = DateTimeFormatter.ofPattern("dd-MM-yyyy_HH-mm-ss")
    val formattedDate = dateFormat.format(date)
    predictions.coalesce(1)
      .write
      .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false")
      .option("header","true")
      .format("csv")
      .save("result/predictions_" + formattedDate + ".csv")
  }

  /**
   *
   * @param spark            : spark session
   * @param trainingDataName : name of the training dataFrame
   * @return : The model loaded if already exists, otherwise the model created.
   */
  private def getModel(spark: SparkSession, trainingDataName: String): PipelineModel = {
    val modelPath = "./spark-lr-model"
    try {
      //if model exists
      PipelineModel.load(modelPath)
    } catch {
      //else build the model
      case e: Exception =>
        // We prepare data to train the model
        val trainingData = prepareData(spark, trainingDataName)
        // Index labels, adding metadata to the label column.
        val labelIndexer = new StringIndexer()
          .setInputCol("label")
          .setOutputCol("indexedLabel")
          .fit(trainingData)

        //We define a StringIndexers for the categorical variable media
        val indexerMedia = new StringIndexer()
          .setInputCol("media")
          .setOutputCol("mediaIndex")

        //We define a StringIndexers for the categorical variable media
        val indexerAppOrSite = new StringIndexer()
          .setInputCol("appOrSite")
          .setOutputCol("appOrSiteIndex")

        // Create features column
        val assembler = new VectorAssembler()
          .setInputCols(Array("bidFloor", "mediaIndex", "appOrSiteIndex", "area",
            "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
            "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
            "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26"))
          .setOutputCol("features")

        val lr = new LogisticRegression()
          .setWeightCol("classWeightCol")
          .setLabelCol("indexedLabel")
          .setFeaturesCol("features")


        // Convert indexed labels back to original labels.
        val labelConverter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("predictedLabel")
          .setLabels(labelIndexer.labels)

        // Chain indexers and tree in a Pipeline
        val pipeline = new Pipeline().setStages(Array(labelIndexer, indexerMedia, indexerAppOrSite, assembler, lr, labelConverter))

        // Train model.  This also runs the indexers.
        val model = pipeline.fit(trainingData)
        model.write.overwrite().save(modelPath)
        model
    }
  }

  /**
   *
   * @param spark : spark session
   * @param data  : name of the dataFrame to prepare
   * @return : the dataFrame prepared
   */
  private def prepareData(spark: SparkSession, data: String): DataFrame = {
    // Read json data
    val jsonData = spark.read.json(data)

    // DATA CLEANING
    val cleanedData = Cleaner.generalClean(jsonData)

    // Select only the needed columns
    cleanedData.select(
      "label", "classWeightCol",
      "bidFloor", "media", "appOrSite", "area",
      "IAB1", "IAB2", "IAB3", "IAB4", "IAB5", "IAB6", "IAB7", "IAB8", "IAB9", "IAB10",
      "IAB11", "IAB12", "IAB13", "IAB14", "IAB15", "IAB16", "IAB17", "IAB18", "IAB19",
      "IAB20", "IAB21", "IAB22", "IAB23", "IAB24", "IAB25", "IAB26")

  }

}
