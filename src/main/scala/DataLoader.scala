import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{FloatType, Metadata, StructField, StructType}

/**
  * Created by marti on 25/03/2017.
  */

/**
  * Load iris data.
  *
  * The iris data is a collection of data collected by R.A. Fisher. It has measurements of various iris flowers
  * and is widely used for beginner statistics and machine-learning problems.
  *
  * The data is a CSV with no header. It is in the format:
  * sepal length in cm, sepal width in cm, petal length in cm, petal width in cm, iris type
  *
  * Example:
  * 5.1,3.5,1.4,0.2,Iris-setosa
  *
  * @return a Dataframe with two columns. `irisFeatureColumn` contains the feature `Vector`s and `irisTypeColumn` contains the `String` iris types
  */

trait DataLoader {

def loadIris(filePath: String, spark: SparkSession): DataFrame = {
    val schema = new StructType()
                .add("x1", FloatType, false)
                .add("x2", FloatType, false)
                .add("x3", FloatType, false)
                .add("x4", FloatType, false)
                .add("y", FloatType, false)

    val assembler = new VectorAssembler()
                    .setInputCols(Array("x1", "x2", "x3", "x4"))
                    .setOutputCol("features")

    val dataset = spark.read.schema(schema).csv(filePath)

    assembler.transform(dataset)
  }
}
