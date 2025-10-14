====================================================================================================
  PARQUET DATA ANALYSIS REPORT
====================================================================================================
Base Directory: .
Generated: 2025-10-15 00:12:17.226971

====================================================================================================
  SECTION 1: ROOT DIRECTORY PARQUET FILES
====================================================================================================
Found 5 parquet files in root directory:

  - apt.parquet
  - flightlist_rank.parquet
  - flightlist_train.parquet
  - fuel_rank_submission.parquet
  - fuel_train.parquet

====================================================================================================
  FILE: ./apt.parquet
====================================================================================================

📊 BASIC INFORMATION
  File Size: 0.14 MB
  Shape: 8,787 rows × 4 columns
  Memory Usage: 0.65 MB
  Number of Row Groups: 1

📋 SCHEMA
  Column Name                    | Data Type          | Non-Null Count      | Null Count
  -----------------------------------------------------------------------------------------------
  icao                           | object             |           8,787 | 0
  longitude                      | float64            |           8,787 | 0
  latitude                       | float64            |           8,787 | 0
  elevation                      | float64            |           8,568 | 219

📈 NUMERIC COLUMN STATISTICS
         longitude     latitude     elevation
count  8787.000000  8787.000000   8568.000000
mean      5.978341    20.075946   1223.274160
std      88.850548    28.571015   1751.824411
min    -179.876999   -79.777778  -1266.000000
25%     -77.067030    -4.082305     89.000000
50%      12.468900    27.335199    527.500000
75%      76.432768    43.075251   1538.250000
max     179.975700    82.517799  14472.000000

🏷️  CATEGORICAL COLUMN SUMMARY
  icao: 8,787 unique values

📄 SAMPLE DATA (First 5 rows)
   icao  longitude   latitude  elevation
0  HCAD  46.637500   6.095802     1001.0
1  OAAK  65.206746  36.942659      900.0
2  OALG  69.023102  33.986500     6333.0
3  OAMK  67.848602  32.879601     6600.0
4  OAPJ  67.020445  34.391726     8800.0

✅ DATA QUALITY
  Total Missing Values: 219 (0.62%)
  Duplicate Rows: 0
  Columns with Missing Values:
    - elevation: 219 (2.49%)


====================================================================================================
  FILE: ./flightlist_rank.parquet
====================================================================================================

📊 BASIC INFORMATION
  File Size: 0.07 MB
  Shape: 1,888 rows × 9 columns
  Memory Usage: 0.80 MB
  Number of Row Groups: 1

📋 SCHEMA
  Column Name                    | Data Type          | Non-Null Count      | Null Count
  -----------------------------------------------------------------------------------------------
  flight_date                    | object             |           1,888 | 0
  aircraft_type                  | object             |           1,888 | 0
  takeoff                        | datetime64[us]     |           1,888 | 0
  landed                         | datetime64[us]     |           1,888 | 0
  origin_icao                    | object             |           1,888 | 0
  origin_name                    | object             |           1,888 | 0
  destination_icao               | object             |           1,888 | 0
  destination_name               | object             |           1,888 | 0
  flight_id                      | object             |           1,888 | 0

🏷️  CATEGORICAL COLUMN SUMMARY
  flight_date: 27 unique values
  aircraft_type: 19 unique values
  origin_icao: 138 unique values
  origin_name: 138 unique values
  destination_icao: 136 unique values
  destination_name: 136 unique values
  flight_id: 1,888 unique values

📄 SAMPLE DATA (First 5 rows)
  flight_date aircraft_type             takeoff              landed origin_icao                         origin_name destination_icao                                destination_name     flight_id
0  2025-09-01          A320 2025-09-01 00:46:27 2025-09-01 04:13:27        MMGL   Guadalajara International Airport             KOAK                   Oakland International Airport  prc806615763
1  2025-09-01          A20N 2025-09-01 00:40:28 2025-09-01 04:36:17        MSLP  San Salvador International Airport             KIAD         Washington Dulles International Airport  prc806615831
2  2025-09-01          A320 2025-09-01 00:58:47 2025-09-01 04:44:01        MMMM       Morelia International Airport             KOAK                   Oakland International Airport  prc806618686
3  2025-09-01          A20N 2025-09-01 01:11:08 2025-09-01 07:19:51        SEGU     Guayaquil International Airport             KJFK  New York John F. Kennedy International Airport  prc806619863
4  2025-09-01          A320 2025-09-01 01:13:38 2025-09-01 05:04:01        MMMM       Morelia International Airport             KSMF                Sacramento International Airport  prc806620863

✅ DATA QUALITY
  Total Missing Values: 0 (0.00%)
  Duplicate Rows: 0


====================================================================================================
  FILE: ./flightlist_train.parquet
====================================================================================================

📊 BASIC INFORMATION
  File Size: 0.34 MB
  Shape: 11,037 rows × 9 columns
  Memory Usage: 4.66 MB
  Number of Row Groups: 1

📋 SCHEMA
  Column Name                    | Data Type          | Non-Null Count      | Null Count
  -----------------------------------------------------------------------------------------------
  flight_date                    | object             |          11,037 | 0
  aircraft_type                  | object             |          11,037 | 0
  takeoff                        | datetime64[us]     |          11,037 | 0
  landed                         | datetime64[us]     |          11,037 | 0
  origin_icao                    | object             |          11,037 | 0
  origin_name                    | object             |          11,037 | 0
  destination_icao               | object             |          11,037 | 0
  destination_name               | object             |          11,037 | 0
  flight_id                      | object             |          11,037 | 0

🏷️  CATEGORICAL COLUMN SUMMARY
  flight_date: 124 unique values
  aircraft_type: 26 unique values
  origin_icao: 273 unique values
  origin_name: 273 unique values
  destination_icao: 260 unique values
  destination_name: 261 unique values
  flight_id: 11,037 unique values

📄 SAMPLE DATA (First 5 rows)
  flight_date aircraft_type             takeoff              landed origin_icao                            origin_name destination_icao                             destination_name     flight_id
0  2025-04-12          B789 2025-04-12 16:15:35 2025-04-13 04:33:52        WMKK     Kuala Lumpur International Airport             EHAM                   Amsterdam Schiphol Airport  prc770822360
1  2025-04-12          A359 2025-04-12 16:40:41 2025-04-13 03:10:15        ZSPD  Shanghai Pudong International Airport             NZAA                             Auckland Airport  prc770831136
2  2025-04-12          A359 2025-04-12 17:01:38 2025-04-13 06:10:27        ZSPD  Shanghai Pudong International Airport             LEMD                       Madrid Barajas Airport  prc770835414
3  2025-04-12          B789 2025-04-12 17:53:11 2025-04-13 05:05:13        ZSPD  Shanghai Pudong International Airport             LHBP  Budapest Ferenc Liszt International Airport  prc770844923
4  2025-04-12          B789 2025-04-12 18:05:27 2025-04-13 06:23:52        ZSPD  Shanghai Pudong International Airport             LFML                   Marseille Provence Airport  prc770847190

✅ DATA QUALITY
  Total Missing Values: 0 (0.00%)
  Duplicate Rows: 0


====================================================================================================
  FILE: ./fuel_rank_submission.parquet
====================================================================================================

📊 BASIC INFORMATION
  File Size: 0.61 MB
  Shape: 24,289 rows × 5 columns
  Memory Usage: 2.52 MB
  Number of Row Groups: 1

📋 SCHEMA
  Column Name                    | Data Type          | Non-Null Count      | Null Count
  -----------------------------------------------------------------------------------------------
  idx                            | int64              |          24,289 | 0
  flight_id                      | object             |          24,289 | 0
  start                          | datetime64[ns]     |          24,289 | 0
  end                            | datetime64[ns]     |          24,289 | 0
  fuel_kg                        | object             |               0 | 24,289

📈 NUMERIC COLUMN STATISTICS
               idx
count  24289.00000
mean   12144.00000
std     7011.77468
min        0.00000
25%     6072.00000
50%    12144.00000
75%    18216.00000
max    24288.00000

🏷️  CATEGORICAL COLUMN SUMMARY
  flight_id: 1,888 unique values
  fuel_kg: 0 unique values
    Values: [None]

📄 SAMPLE DATA (First 5 rows)
   idx     flight_id                   start                     end fuel_kg
0    0  prc806615763 2025-09-01 03:03:10.925 2025-09-01 03:07:51.584    None
1    1  prc806615763 2025-09-01 03:07:51.584 2025-09-01 03:12:50.921    None
2    2  prc806615763 2025-09-01 03:12:50.921 2025-09-01 03:17:51.404    None
3    3  prc806615763 2025-09-01 03:17:51.404 2025-09-01 03:22:50.539    None
4    4  prc806615763 2025-09-01 03:22:50.539 2025-09-01 03:27:50.727    None

✅ DATA QUALITY
  Total Missing Values: 24,289 (20.00%)
  Duplicate Rows: 0
  Columns with Missing Values:
    - fuel_kg: 24,289 (100.00%)


====================================================================================================
  FILE: ./fuel_train.parquet
====================================================================================================

📊 BASIC INFORMATION
  File Size: 3.54 MB
  Shape: 131,530 rows × 5 columns
  Memory Usage: 11.67 MB
  Number of Row Groups: 1

📋 SCHEMA
  Column Name                    | Data Type          | Non-Null Count      | Null Count
  -----------------------------------------------------------------------------------------------
  idx                            | int64              |         131,530 | 0
  flight_id                      | object             |         131,530 | 0
  start                          | datetime64[ns]     |         131,530 | 0
  end                            | datetime64[ns]     |         131,530 | 0
  fuel_kg                        | float64            |         131,530 | 0

📈 NUMERIC COLUMN STATISTICS
                 idx        fuel_kg
count  131530.000000  131530.000000
mean    65764.500000     557.705484
std     37969.584791     883.558032
min         0.000000       0.453592
25%     32882.250000     100.000000
50%     65764.500000     200.000000
75%     98646.750000     599.000000
max    131529.000000   32205.032000

🏷️  CATEGORICAL COLUMN SUMMARY
  flight_id: 11,037 unique values

📄 SAMPLE DATA (First 5 rows)
   idx     flight_id                   start                     end  fuel_kg
0    0  prc770822360 2025-04-13 02:31:04.447 2025-04-13 03:01:04.487   2500.0
1    1  prc770822360 2025-04-13 03:01:04.487 2025-04-13 03:16:04.226   1200.0
2    2  prc770822360 2025-04-13 03:16:04.458 2025-04-13 03:46:04.417   2500.0
3    3  prc770822360 2025-04-13 03:46:04.417 2025-04-13 04:01:04.540   1300.0
4    4  prc770822360 2025-04-13 04:01:04.540 2025-04-13 04:16:04.263    600.0

✅ DATA QUALITY
  Total Missing Values: 0 (0.00%)
  Duplicate Rows: 0


====================================================================================================
  SECTION 2: FLIGHTS_RANK DIRECTORY (SAMPLE)
====================================================================================================
Directory contains 1,888 parquet files
Analyzing SAMPLE file (first alphabetically): prc806615763.parquet

====================================================================================================
  FILE: ./flights_rank/prc806615763.parquet
====================================================================================================

📊 BASIC INFORMATION
  File Size: 0.25 MB
  Shape: 9,462 rows × 13 columns
  Memory Usage: 2.23 MB
  Number of Row Groups: 1

📋 SCHEMA
  Column Name                    | Data Type          | Non-Null Count      | Null Count
  -----------------------------------------------------------------------------------------------
  timestamp                      | datetime64[ns]     |           9,462 | 0
  flight_id                      | object             |           9,462 | 0
  typecode                       | object             |           9,462 | 0
  latitude                       | float64            |           9,462 | 0
  longitude                      | float64            |           9,462 | 0
  altitude                       | float64            |           9,462 | 0
  groundspeed                    | float64            |           5,984 | 3,478
  track                          | float64            |           5,984 | 3,478
  vertical_rate                  | float64            |           5,984 | 3,478
  mach                           | float64            |              12 | 9,450
  TAS                            | float64            |               0 | 9,462
  CAS                            | float64            |               0 | 9,462
  source                         | object             |           9,462 | 0

📈 NUMERIC COLUMN STATISTICS
          latitude    longitude      altitude  groundspeed        track  vertical_rate       mach  TAS  CAS
count  9462.000000  9462.000000   9462.000000  5984.000000  5984.000000    5984.000000  12.000000  0.0  0.0
mean     34.664061  -119.115716  28371.207144   365.508523   317.280406    -697.775401   0.718667  NaN  NaN
std       2.332466     1.988110  12280.717676   110.389694    12.918838     807.934454   0.069255  NaN  NaN
min      29.866425  -122.225348    -25.000000    98.000000   277.488899   -2496.000000   0.526000  NaN  NaN
25%      32.561124  -120.948292  21506.250000   313.000000   311.803578   -1536.000000   0.742000  NaN  NaN
50%      35.010914  -119.089108  36000.000000   431.000000   318.122130    -448.000000   0.746000  NaN  NaN
75%      36.855290  -117.314637  36000.000000   446.000000   329.345232       0.000000   0.749250  NaN  NaN
max      37.708895  -114.969547  36050.000000   457.000000   331.570951     256.000000   0.751000  NaN  NaN

🏷️  CATEGORICAL COLUMN SUMMARY
  flight_id: 1 unique values
    Values: ['prc806615763']
  typecode: 1 unique values
    Values: ['A320']
  source: 2 unique values
    Values: ['adsb', 'acars']

📄 SAMPLE DATA (First 5 rows)
                      timestamp     flight_id typecode   latitude   longitude  altitude  groundspeed       track  vertical_rate  mach  TAS  CAS source
0 2025-09-01 02:42:49.049999952  prc806615763     A320  29.866425 -114.969547   36025.0        441.0  308.740964            0.0   NaN  NaN  NaN   adsb
1 2025-09-01 02:42:50.621999979  prc806615763     A320  29.869034 -114.973297   36025.0        440.0  308.639491            0.0   NaN  NaN  NaN   adsb
2 2025-09-01 02:42:51.713999987  prc806615763     A320  29.870361 -114.975199   36025.0        440.0  308.639491            0.0   NaN  NaN  NaN   adsb
3 2025-09-01 02:42:52.816999912  prc806615763     A320  29.871689 -114.977100   36025.0        440.0  308.639491            0.0   NaN  NaN  NaN   adsb
4 2025-09-01 02:45:45.769000053  prc806615763     A320  30.091739 -115.294098   35975.0        442.0  308.477596           64.0   NaN  NaN  NaN   adsb

✅ DATA QUALITY
  Total Missing Values: 38,808 (31.55%)
  Duplicate Rows: 0
  Columns with Missing Values:
    - groundspeed: 3,478 (36.76%)
    - track: 3,478 (36.76%)
    - vertical_rate: 3,478 (36.76%)
    - mach: 9,450 (99.87%)
    - TAS: 9,462 (100.00%)
    - CAS: 9,462 (100.00%)


====================================================================================================
  SECTION 3: FLIGHTS_TRAIN DIRECTORY (SAMPLE)
====================================================================================================
Directory contains 11,037 parquet files
Analyzing SAMPLE file (first alphabetically): prc770822360.parquet

====================================================================================================
  FILE: ./flights_train/prc770822360.parquet
====================================================================================================

📊 BASIC INFORMATION
  File Size: 0.62 MB
  Shape: 25,461 rows × 13 columns
  Memory Usage: 6.00 MB
  Number of Row Groups: 1

📋 SCHEMA
  Column Name                    | Data Type          | Non-Null Count      | Null Count
  -----------------------------------------------------------------------------------------------
  timestamp                      | datetime64[ns]     |          25,461 | 0
  flight_id                      | object             |          25,461 | 0
  typecode                       | object             |          25,461 | 0
  latitude                       | float64            |          25,461 | 0
  longitude                      | float64            |          25,461 | 0
  altitude                       | float64            |          25,453 | 8
  groundspeed                    | float64            |          18,078 | 7,383
  track                          | float64            |          18,078 | 7,383
  vertical_rate                  | float64            |          18,078 | 7,383
  mach                           | float64            |               8 | 25,453
  TAS                            | float64            |               0 | 25,461
  CAS                            | float64            |               0 | 25,461
  source                         | object             |          25,461 | 0

📈 NUMERIC COLUMN STATISTICS
           latitude     longitude      altitude   groundspeed         track  vertical_rate      mach  TAS  CAS
count  25461.000000  25461.000000  25453.000000  18078.000000  18078.000000    18078.00000  8.000000  0.0  0.0
mean      43.544574     31.031522  32209.124233    446.802191    288.585379     -157.53247  0.792500  NaN  NaN
std        9.852854     24.412068   9139.500677     81.660508     33.211905      692.26873  0.124986  NaN  NaN
min        2.791214      4.513184    349.910400     96.000000     89.570826    -3840.00000  0.590000  NaN  NaN
25%       41.520538     13.170584  34000.135200    459.000000    274.398705      -64.00000  0.792500  NaN  NaN
50%       45.878815     22.759519  35974.934400    472.000000    299.357754      -64.00000  0.860000  NaN  NaN
75%       50.038239     43.305484  35999.928000    485.000000    304.369701        0.00000  0.860000  NaN  NaN
max       52.688335    101.692991  36024.921600    525.000000    342.424814     2688.00000  0.860000  NaN  NaN

🏷️  CATEGORICAL COLUMN SUMMARY
  flight_id: 1 unique values
    Values: ['prc770822360']
  typecode: 1 unique values
    Values: ['B789']
  source: 2 unique values
    Values: ['adsb', 'acars']

📄 SAMPLE DATA (First 5 rows)
                      timestamp     flight_id typecode  latitude   longitude   altitude  groundspeed      track  vertical_rate  mach  TAS  CAS source
0 2025-04-12 16:16:35.464999914  prc770822360     B789  2.791214  101.692991  1274.9784        193.0  325.98065         2624.0   NaN  NaN  NaN   adsb
1 2025-04-12 16:16:43.220999956  prc770822360     B789  2.796982  101.689267  1499.9208          NaN        NaN            NaN   NaN  NaN  NaN   adsb
2 2025-04-12 16:17:02.757999897  prc770822360     B789  2.814102  101.681260  1749.8568          NaN        NaN            NaN   NaN  NaN  NaN   adsb
3 2025-04-12 16:17:06.473999977  prc770822360     B789  2.817714  101.680077  1825.1424          NaN        NaN            NaN   NaN  NaN  NaN   adsb
4 2025-04-12 16:17:07.440000057  prc770822360     B789  2.818634  101.679770  1825.1424          NaN        NaN            NaN   NaN  NaN  NaN   adsb

✅ DATA QUALITY
  Total Missing Values: 98,532 (29.77%)
  Duplicate Rows: 27
  Columns with Missing Values:
    - altitude: 8 (0.03%)
    - groundspeed: 7,383 (29.00%)
    - track: 7,383 (29.00%)
    - vertical_rate: 7,383 (29.00%)
    - mach: 25,453 (99.97%)
    - TAS: 25,461 (100.00%)
    - CAS: 25,461 (100.00%)


====================================================================================================
  ANALYSIS SUMMARY
====================================================================================================
Total Files Analyzed: 7
  ✅ Successful: 7
  ❌ Failed: 0
====================================================================================================
Analysis complete!
