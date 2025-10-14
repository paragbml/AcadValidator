# Dataset Schema Documentation

## UGC-AICTE Synthetic Dataset (2019-2023)

### Overview
This dataset contains synthetic but realistic data for 40 higher education institutions in India over a 5-year period (2019-2023). The data is designed to simulate real institutional performance metrics used in UGC and AICTE approval processes.

### Dataset Details
- **Total Records**: 200 (40 institutions × 5 years)
- **Time Period**: 2019-2023
- **Institution Types**: IITs, NITs, Private Universities, State Universities, Engineering Colleges

### Column Descriptions

| Column Name | Data Type | Range | Description |
|-------------|-----------|-------|-------------|
| `College_Name` | String | - | Name of the educational institution |
| `Year` | Integer | 2019-2023 | Academic year |
| `AICTE_Approval_Score` | Float | 40-100 | AICTE approval rating (higher is better) |
| `UGC_Rating` | Float | 3-10 | UGC rating scale (higher is better) |
| `NIRF_Rank` | Integer | 1-200 | National Institutional Ranking Framework rank (lower is better) |
| `Placement_Percentage` | Float | 25-100 | Percentage of students placed in jobs |
| `Faculty_Student_Ratio` | Float | 8-25 | Faculty to student ratio (lower is better) |
| `Research_Projects` | Integer | 0-50 | Number of active research projects |
| `Infrastructure_Score` | Float | 40-100 | Infrastructure quality score (higher is better) |
| `Student_Satisfaction_Score` | Float | 40-100 | Student satisfaction rating (higher is better) |
| `Final_Performance_Index` | Float | 30-100 | **Target Variable** - Overall performance index |

### Data Generation Logic

#### Institution Tiers
Institutions are categorized into three tiers based on performance:

1. **Tier 1 (Elite)**: IITs, IIITs, top NITs
   - Base Performance: 85-95
   - NIRF Rank: 1-50
   - High placement rates (70-100%)

2. **Tier 2 (Good)**: Most NITs, reputed private universities
   - Base Performance: 65-85
   - NIRF Rank: 30-150
   - Moderate placement rates (50-90%)

3. **Tier 3 (Average)**: State universities, newer institutions
   - Base Performance: 30-75
   - NIRF Rank: 100-200
   - Lower placement rates (25-70%)

#### Temporal Trends
- **COVID Impact**: 2020-2021 show slight decline in most metrics
- **Recovery**: 2022-2023 show improvement trends
- **Year-over-year**: Small positive trend (+0.5 points per year)

#### Performance Index Calculation
The Final Performance Index is calculated using weighted average:

```
Performance Index = 
  0.20 × AICTE_Approval_Score +
  0.15 × (UGC_Rating × 10) +
  0.15 × (Inverse_NIRF_Score) +
  0.20 × Placement_Percentage +
  0.05 × (Faculty_Ratio_Score) +
  0.10 × (Research_Projects × 2) +
  0.10 × Infrastructure_Score +
  0.05 × Student_Satisfaction_Score
```

### Data Quality Features
- **Realistic Correlations**: Metrics are correlated as expected in real data
- **Institutional Consistency**: Each institution maintains character across years
- **Missing Data**: No missing values (cleaned dataset)
- **Outlier Handling**: All values within realistic bounds

### Usage Notes
- Use `Final_Performance_Index` as target variable for prediction models
- Institution names are representative but not exhaustive of all Indian institutions
- Data simulates real-world patterns but should not be used for actual institutional assessment
- Suitable for machine learning model training and dashboard development

### Sample Statistics
- Mean Performance Index: ~76.5
- Standard Deviation: ~5.8
- Best Performing: IITs and IIITs (85-90 range)
- Institutional Spread: Good representation across performance spectrum

### File Location
`/data/ugc_aicte_synthetic_dataset_2019_2023.csv`