import FirstSparkApplication.{jsonData, spark}
import org.apache.parquet.filter2.predicate.Operators.Column
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf

object Cleaner {

  /**
   * Clean "interests" column
   * @param df : dataframe containing a colomn "interests"
   * @return the given dataframe updated with new interests columns
   */
  def cleanInterests(df: DataFrame): DataFrame = {

    /** REPLACE NULL WITH N/A */
    /** KEEP ONLY INTERESTS IN THE RIGHT FORMAT (code starting with IAB) */
    val removeIfNotCodeUDF = udf { interests: String =>
      if (interests == null) "N/A"
      else interests.split(",").filter(_.startsWith("IAB")) mkString ","
    }

    val interestsCleaned = removeIfNotCodeUDF(df.col("interests"))

    val newDataframe = df.withColumn("interests", interestsCleaned)

    /** CREATE 26 INTEREST COLUMNS */
    createInterestColumns(newDataframe)
  }


  /**
   * Create 26 columns for each interest class.
   * Each column contains 1 if the row has the interest specified in "interests" column, else contains 0
   * @param df : dataframe containing a column "interests"
   * @return the given dataframe updated with new columns for each interest
   */
  def createInterestColumns(df: DataFrame): DataFrame = {

    @scala.annotation.tailrec
    def loop(accDataframe: DataFrame, interestNb: Int): DataFrame = {

      if (interestNb == 27) accDataframe

      else {
        val IABn = "IAB" + interestNb

        val containsInterestUDF = udf { listInterests: String =>
          if (listInterests.contains(IABn + ",") || listInterests.contains(IABn + "-")) "1"
          else "0"
        }

        val IABnColumn = containsInterestUDF(accDataframe.col("interests"))
        val dfUpdated = accDataframe.withColumn(IABn, IABnColumn)

        loop(dfUpdated, interestNb + 1)
      }
    }

    loop(df, 1)
  }
}