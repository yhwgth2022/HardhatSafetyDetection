import os
import xarray as xr
import spark
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import col

from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("NetCDF Processing") \
    .getOrCreate()

# 定义正确的数据文件路径
folder_path = r'C:\Big_data\Data'

# 获取所有子文件夹中的 .nc 文件
files = []
for root, dirs, file_names in os.walk(folder_path):
    for file in file_names:
        if file.endswith('.nc'):
            files.append(os.path.join(root, file))

# 按创建时间排序文件
files_sorted = sorted(files, key=lambda x: os.path.getctime(x))

# 定义 Spark DataFrame 的 schema
schema = StructType([
    StructField("time", StringType(), True),
    StructField("SST", FloatType(), True),
    StructField("DQF", FloatType(), True)
])

# 定义一个空的 RDD 来存储所有数据
all_data_rdd = spark.sparkContext.emptyRDD()

# 依次处理每个文件
for file in files_sorted:
    print(f"正在处理文件: {os.path.basename(file)}")

    # 打开 NetCDF 文件
    ds = xr.open_dataset(file)

    # 提取相关特征
    time_bounds = ds['time_bounds'].values if 'time_bounds' in ds.variables else None
    sst = ds['SST'].values if 'SST' in ds.variables else None
    dqf = ds['DQF'].values if 'DQF' in ds.variables else None

    # 检查 SST 和 DQF 的长度是否相等
    if sst is not None and dqf is not None and sst.size == dqf.size:
        # 提取 time_bounds 的第一个元素作为时间标记
        if time_bounds is not None and time_bounds.ndim > 0:
            time_bound_name = str(time_bounds[0])  # 提取第一个时间值并将其转换为字符串
        else:
            time_bound_name = "unknown_time"

        # 将 SST 和 DQF 转换为一维数组
        sst_flattened = sst.flatten()
        dqf_flattened = dqf.flatten()

        # 将当前文件的数据转换为 RDD
        file_rdd = spark.sparkContext.parallelize([
            Row(time=time_bound_name, SST=float(sst_val), DQF=float(dqf_val))
            for sst_val, dqf_val in zip(sst_flattened, dqf_flattened)
        ])

        # 将当前文件的 RDD 合并到总的 RDD 中
        all_data_rdd = all_data_rdd.union(file_rdd)

    # 关闭 NetCDF 文件以释放资源
    ds.close()

# 将 RDD 转换为 Spark DataFrame
all_data_df = spark.createDataFrame(all_data_rdd, schema)

# 将时间列转换为日期时间格式
all_data_df = all_data_df.withColumn("time", col("time").cast("timestamp"))

# 按时间列排序
all_data_df = all_data_df.orderBy(col("time"))

# 显示前几行结果
all_data_df.show()

# 保存最终处理后的数据到 Parquet 文件
all_data_df.coalesce(1).write.mode("overwrite").parquet("output/sst_data.parquet")

