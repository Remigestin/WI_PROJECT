import org.apache.spark.sql.SparkSession

object FirstSparkApplication extends App {

  //Only for windows user I think
  System.setProperty("hadoop.home.dir", "C:\\winutils")

  val spark = SparkSession.builder
    .master("local[*]")
    .appName("FirstSparkApplication")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")


  val jsonData = spark.read.json("data-students.json")
  jsonData.printSchema()

  jsonData.createOrReplaceTempView("data")
  val tmp = spark.sql("SELECT * FROM data LIMIT 10")
  tmp.show()

}