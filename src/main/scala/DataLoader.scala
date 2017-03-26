import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{FloatType, Metadata, StructField, StructType}

/**
  * Created by marti on 25/03/2017.
  */
trait DataLoader {


def featuresCol = "X"
def labelCol = "y"


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
