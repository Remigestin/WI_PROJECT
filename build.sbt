name := "WI_project"
version := "0.1"
scalaVersion := "2.12.0"

val sparkVersion = "2.4.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "2.7.1",
  "org.apache.hadoop" % "hadoop-common" % "2.7.1"

)
