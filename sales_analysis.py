# ============================================
# SALES AND PROFIT ANALYSIS - COMPLETE CODE
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# 1. DATA LOADING AND PREPARATION
# ============================================

print("="*50)
print("1. DATA LOADING AND PREPARATION")
print("="*50)

# Load the dataset (assuming it's named 'sales_data.csv')
# For demonstration, I'll create a sample dataset based on your structure
np.random.seed(42)

# Generate sample data
dates = pd.date_range(start='2016-01-01', end='2016-12-31', freq='D')
sample_size = 1000

data = {
    'Row ID': range(1, sample_size + 1),
    'Order ID': [f'CA-2016-{100000 + i}' for i in range(sample_size)],
    'Order Date': np.random.choice(dates, sample_size),
    'Ship Date': np.random.choice(dates, sample_size),
    'Ship Mode': np.random.choice(['Second Class', 'Standard Class', 'First Class', 'Same Day'], 
                                   sample_size, p=[0.3, 0.4, 0.2, 0.1]),
    'Customer ID': [f'CG-{np.random.randint(10000, 99999)}' for _ in range(sample_size)],
    'Customer Name': ['Customer_' + str(i) for i in range(sample_size)],
    'Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], 
                                sample_size, p=[0.5, 0.3, 0.2]),
    'Country': 'United States',
    'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], sample_size),
    'State': np.random.choice(['California', 'Texas', 'New York', 'Florida', 'Illinois'], sample_size),
    'Postal Code': np.random.randint(10000, 99999, sample_size),
    'Region': np.random.choice(['South', 'West', 'Central', 'East'], sample_size),
    'Product ID': [f'PROD-{np.random.randint(1000, 9999)}' for _ in range(sample_size)],
    'Category': np.random.choice(['Furniture', 'Office Supplies', 'Technology'], 
                                 sample_size, p=[0.35, 0.35, 0.3]),
    'Sub-Category': np.random.choice(['Bookcases', 'Chairs', 'Tables', 'Phones', 'Accessories', 
                                      'Binders', 'Paper', 'Storage', 'Machines'], sample_size),
    'Product Name': ['Product_' + str(i) for i in range(sample_size)],
    'Sales': np.random.uniform(10, 2000, sample_size).round(2),
    'Quantity': np.random.randint(1, 10, sample_size),
    'Discount': np.random.uniform(0, 0.5, sample_size).round(2),
    'Profit': np.random.uniform(-50, 500, sample_size).round(2)
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert dates to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Extract month and month name
df['Order Month'] = df['Order Date'].dt.month
df['Order Month Name'] = df['Order Date'].dt.strftime('%B')
df['Order Year'] = df['Order Date'].dt.year

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData Info:")
print(df.info())
print(f"\nBasic Statistics:")
print(df[['Sales', 'Quantity', 'Discount', 'Profit']].describe())

# ============================================
# 2. MONTHLY SALES ANALYSIS
# ============================================

print("\n" + "="*50)
print("2. MONTHLY SALES ANALYSIS")
print("="*50)

# Calculate monthly sales
monthly_sales = df.groupby(['Order Year', 'Order Month', 'Order Month Name']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'  # Number of orders
}).rename(columns={'Order ID': 'Order Count'}).reset_index()

# Sort by month
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_sales['Order Month Name'] = pd.Categorical(monthly_sales['Order Month Name'], 
                                                    categories=month_order, ordered=True)
monthly_sales = monthly_sales.sort_values('Order Month')

# Identify best and worst months
max_sales_month = monthly_sales.loc[monthly_sales['Sales'].idxmax()]
min_sales_month = monthly_sales.loc[monthly_sales['Sales'].idxmin()]

print(f"\nHighest Sales Month: {max_sales_month['Order Month Name']}")
print(f"  Total Sales: ${max_sales_month['Sales']:,.2f}")
print(f"  Total Profit: ${max_sales_month['Profit']:,.2f}")
print(f"  Number of Orders: {max_sales_month['Order Count']}")

print(f"\nLowest Sales Month: {min_sales_month['Order Month Name']}")
print(f"  Total Sales: ${min_sales_month['Sales']:,.2f}")
print(f"  Total Profit: ${min_sales_month['Profit']:,.2f}")
print(f"  Number of Orders: {min_sales_month['Order Count']}")

# Visualization 1: Monthly Sales Trend
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Monthly Sales and Profit Analysis', fontsize=16, fontweight='bold')

# Plot 1: Monthly Sales
axes[0, 0].bar(monthly_sales['Order Month Name'], monthly_sales['Sales'], 
               color='skyblue', edgecolor='navy')
axes[0, 0].set_title('Monthly Total Sales', fontweight='bold')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].axhline(y=monthly_sales['Sales'].mean(), color='r', linestyle='--', 
                   label=f"Avg: ${monthly_sales['Sales'].mean():,.0f}")
axes[0, 0].legend()

# Plot 2: Monthly Profit
axes[0, 1].bar(monthly_sales['Order Month Name'], monthly_sales['Profit'], 
               color='lightgreen', edgecolor='darkgreen')
axes[0, 1].set_title('Monthly Total Profit', fontweight='bold')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Profit ($)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].axhline(y=monthly_sales['Profit'].mean(), color='r', linestyle='--', 
                   label=f"Avg: ${monthly_sales['Profit'].mean():,.0f}")
axes[0, 1].legend()

# Plot 3: Sales vs Profit correlation
axes[1, 0].scatter(monthly_sales['Sales'], monthly_sales['Profit'], 
                   alpha=0.6, s=100, color='purple')
axes[1, 0].set_title('Sales vs Profit Correlation', fontweight='bold')
axes[1, 0].set_xlabel('Sales ($)')
axes[1, 0].set_ylabel('Profit ($)')
for i, row in monthly_sales.iterrows():
    axes[1, 0].annotate(row['Order Month Name'][:3], 
                       (row['Sales'], row['Profit']),
                       fontsize=9)

# Plot 4: Number of Orders per Month
axes[1, 1].plot(monthly_sales['Order Month Name'], monthly_sales['Order Count'], 
                marker='o', linewidth=2, markersize=8, color='orange')
axes[1, 1].fill_between(monthly_sales['Order Month Name'], 
                        monthly_sales['Order Count'], alpha=0.3, color='orange')
axes[1, 1].set_title('Number of Orders per Month', fontweight='bold')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Number of Orders')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ============================================
# 3. CATEGORY ANALYSIS
# ============================================

print("\n" + "="*50)
print("3. CATEGORY ANALYSIS")
print("="*50)

# Calculate category performance
category_performance = df.groupby('Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Quantity': 'sum'
}).rename(columns={'Order ID': 'Order Count'}).reset_index()

# Calculate profit margin
category_performance['Profit Margin %'] = (category_performance['Profit'] / category_performance['Sales'] * 100).round(2)
category_performance['Avg Order Value'] = (category_performance['Sales'] / category_performance['Order Count']).round(2)

# Identify best and worst categories
max_sales_category = category_performance.loc[category_performance['Sales'].idxmax()]
min_sales_category = category_performance.loc[category_performance['Sales'].idxmin()]
max_profit_category = category_performance.loc[category_performance['Profit'].idxmax()]
max_margin_category = category_performance.loc[category_performance['Profit Margin %'].idxmax()]

print("\nCategory Performance Summary:")
print(category_performance.to_string(index=False))

print(f"\nHighest Sales Category: {max_sales_category['Category']}")
print(f"  Total Sales: ${max_sales_category['Sales']:,.2f}")
print(f"  Profit Margin: {max_sales_category['Profit Margin %']}%")

print(f"\nLowest Sales Category: {min_sales_category['Category']}")
print(f"  Total Sales: ${min_sales_category['Sales']:,.2f}")
print(f"  Profit Margin: {min_sales_category['Profit Margin %']}%")

print(f"\nHighest Profit Category: {max_profit_category['Category']}")
print(f"  Total Profit: ${max_profit_category['Profit']:,.2f}")

print(f"\nHighest Margin Category: {max_margin_category['Category']}")
print(f"  Profit Margin: {max_margin_category['Profit Margin %']}%")

# Visualization 2: Category Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Product Category Analysis', fontsize=16, fontweight='bold')

# Plot 1: Sales by Category
axes[0, 0].bar(category_performance['Category'], category_performance['Sales'], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
axes[0, 0].set_title('Total Sales by Category', fontweight='bold')
axes[0, 0].set_xlabel('Category')
axes[0, 0].set_ylabel('Sales ($)')
for i, v in enumerate(category_performance['Sales']):
    axes[0, 0].text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Profit by Category
axes[0, 1].bar(category_performance['Category'], category_performance['Profit'], 
               color=['#96CEB4', '#FFEEAD', '#D9534F'])
axes[0, 1].set_title('Total Profit by Category', fontweight='bold')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Profit ($)')
for i, v in enumerate(category_performance['Profit']):
    axes[0, 1].text(i, v, f'${v:,.0f}', ha='center', va='bottom' if v >= 0 else 'top', 
                    fontweight='bold', color='red' if v < 0 else 'black')

# Plot 3: Profit Margin by Category
axes[1, 0].bar(category_performance['Category'], category_performance['Profit Margin %'], 
               color=['#588C7E', '#F2E394', '#F2AE72'])
axes[1, 0].set_title('Profit Margin by Category (%)', fontweight='bold')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Profit Margin %')
axes[1, 0].axhline(y=0, color='black', linewidth=0.8)
for i, v in enumerate(category_performance['Profit Margin %']):
    axes[1, 0].text(i, v, f'{v}%', ha='center', va='bottom' if v >= 0 else 'top', 
                    fontweight='bold', color='red' if v < 0 else 'black')

# Plot 4: Quantity Sold by Category
axes[1, 1].pie(category_performance['Quantity'], labels=category_performance['Category'], 
               autopct='%1.1f%%', colors=['#FF9999', '#66B2FF', '#99FF99'])
axes[1, 1].set_title('Quantity Distribution by Category', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================
# 4. SUB-CATEGORY ANALYSIS
# ============================================

print("\n" + "="*50)
print("4. SUB-CATEGORY ANALYSIS")
print("="*50)

# Calculate sub-category performance
subcategory_performance = df.groupby(['Category', 'Sub-Category']).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).rename(columns={'Order ID': 'Order Count'}).reset_index()

# Calculate profit margin
subcategory_performance['Profit Margin %'] = (subcategory_performance['Profit'] / subcategory_performance['Sales'] * 100).round(2)

# Sort by sales
subcategory_performance = subcategory_performance.sort_values('Sales', ascending=False)

print("\nTop 5 Sub-Categories by Sales:")
print(subcategory_performance.head(5).to_string(index=False))

print("\nBottom 5 Sub-Categories by Sales:")
print(subcategory_performance.tail(5).to_string(index=False))

# Visualization 3: Sub-Category Analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Sub-Category Analysis', fontsize=16, fontweight='bold')

# Plot 1: Top 10 Sub-Categories by Sales
top_10_subcat = subcategory_performance.head(10).sort_values('Sales')
axes[0].barh(top_10_subcat['Sub-Category'], top_10_subcat['Sales'], 
             color=plt.cm.Set3(np.arange(len(top_10_subcat))))
axes[0].set_title('Top 10 Sub-Categories by Sales', fontweight='bold')
axes[0].set_xlabel('Sales ($)')
for i, v in enumerate(top_10_subcat['Sales']):
    axes[0].text(v, i, f' ${v:,.0f}', va='center', fontsize=9)

# Plot 2: Profit Margin by Sub-Category
subcategory_performance_sorted = subcategory_performance.sort_values('Profit Margin %', ascending=False)
top_bottom = pd.concat([subcategory_performance_sorted.head(5), subcategory_performance_sorted.tail(5)])
colors = ['green' if x >= 0 else 'red' for x in top_bottom['Profit Margin %']]
axes[1].barh(top_bottom['Sub-Category'], top_bottom['Profit Margin %'], color=colors)
axes[1].set_title('Top & Bottom 5 Sub-Categories by Profit Margin', fontweight='bold')
axes[1].set_xlabel('Profit Margin %')
axes[1].axvline(x=0, color='black', linewidth=0.8)
for i, v in enumerate(top_bottom['Profit Margin %']):
    axes[1].text(v, i, f' {v}%', va='center', fontsize=9, 
                fontweight='bold', color='white' if abs(v) > 5 else 'black')

plt.tight_layout()
plt.show()

# ============================================
# 5. MONTHLY PROFIT ANALYSIS
# ============================================

print("\n" + "="*50)
print("5. MONTHLY PROFIT ANALYSIS")
print("="*50)

# Already calculated in monthly_sales
max_profit_month = monthly_sales.loc[monthly_sales['Profit'].idxmax()]
min_profit_month = monthly_sales.loc[monthly_sales['Profit'].idxmin()]

print(f"\nHighest Profit Month: {max_profit_month['Order Month Name']}")
print(f"  Total Profit: ${max_profit_month['Profit']:,.2f}")
print(f"  Total Sales: ${max_profit_month['Sales']:,.2f}")
print(f"  Profit Margin: {(max_profit_month['Profit']/max_profit_month['Sales']*100):.1f}%")

print(f"\nLowest Profit Month: {min_profit_month['Order Month Name']}")
print(f"  Total Profit: ${min_profit_month['Profit']:,.2f}")
print(f"  Total Sales: ${min_profit_month['Sales']:,.2f}")
print(f"  Profit Margin: {(min_profit_month['Profit']/min_profit_month['Sales']*100):.1f}%")

# ============================================
# 6. CUSTOMER SEGMENT ANALYSIS
# ============================================

print("\n" + "="*50)
print("6. CUSTOMER SEGMENT ANALYSIS")
print("="*50)

segment_performance = df.groupby('Segment').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count',
    'Customer ID': 'nunique'
}).rename(columns={'Order ID': 'Order Count', 'Customer ID': 'Unique Customers'}).reset_index()

segment_performance['Avg Order Value'] = (segment_performance['Sales'] / segment_performance['Order Count']).round(2)
segment_performance['Profit per Customer'] = (segment_performance['Profit'] / segment_performance['Unique Customers']).round(2)
segment_performance['Profit Margin %'] = (segment_performance['Profit'] / segment_performance['Sales'] * 100).round(2)

print("\nCustomer Segment Performance:")
print(segment_performance.to_string(index=False))

# Visualization 4: Customer Segment Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Segment Analysis', fontsize=16, fontweight='bold')

# Plot 1: Sales by Segment
axes[0, 0].pie(segment_performance['Sales'], labels=segment_performance['Segment'], 
               autopct='%1.1f%%', colors=['#FFD700', '#C0C0C0', '#CD7F32'])
axes[0, 0].set_title('Sales Distribution by Segment', fontweight='bold')

# Plot 2: Profit by Segment
colors_seg = ['green' if x >= 0 else 'red' for x in segment_performance['Profit']]
axes[0, 1].bar(segment_performance['Segment'], segment_performance['Profit'], color=colors_seg)
axes[0, 1].set_title('Total Profit by Segment', fontweight='bold')
axes[0, 1].set_xlabel('Segment')
axes[0, 1].set_ylabel('Profit ($)')
for i, v in enumerate(segment_performance['Profit']):
    axes[0, 1].text(i, v, f'${v:,.0f}', ha='center', va='bottom' if v >= 0 else 'top', 
                    fontweight='bold')

# Plot 3: Average Order Value by Segment
axes[1, 0].bar(segment_performance['Segment'], segment_performance['Avg Order Value'], 
               color=['#9ACD32', '#20B2AA', '#FFA07A'])
axes[1, 0].set_title('Average Order Value by Segment', fontweight='bold')
axes[1, 0].set_xlabel('Segment')
axes[1, 0].set_ylabel('Average Order Value ($)')
for i, v in enumerate(segment_performance['Avg Order Value']):
    axes[1, 0].text(i, v, f'${v:.0f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Profit Margin by Segment
axes[1, 1].bar(segment_performance['Segment'], segment_performance['Profit Margin %'], 
               color=['#32CD32', '#1E90FF', '#FF6347'])
axes[1, 1].set_title('Profit Margin by Segment (%)', fontweight='bold')
axes[1, 1].set_xlabel('Segment')
axes[1, 1].set_ylabel('Profit Margin %')
axes[1, 1].axhline(y=0, color='black', linewidth=0.8)
for i, v in enumerate(segment_performance['Profit Margin %']):
    axes[1, 1].text(i, v, f'{v}%', ha='center', va='bottom' if v >= 0 else 'top', 
                    fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================
# 7. SALES TO PROFIT RATIO ANALYSIS
# ============================================

print("\n" + "="*50)
print("7. SALES TO PROFIT RATIO ANALYSIS")
print("="*50)

# Calculate overall ratio
overall_ratio = (df['Profit'].sum() / df['Sales'].sum() * 100).round(2)
print(f"\nOverall Sales to Profit Ratio: {overall_ratio}%")

# Calculate ratio by month
monthly_sales['Profit Ratio %'] = (monthly_sales['Profit'] / monthly_sales['Sales'] * 100).round(2)

# Calculate ratio by category
category_performance['Profit Ratio %'] = category_performance['Profit Margin %']

# Calculate ratio by segment
segment_performance['Profit Ratio %'] = segment_performance['Profit Margin %']

print(f"\nMonthly Profit Ratios:")
for _, row in monthly_sales.iterrows():
    print(f"  {row['Order Month Name']}: {row['Profit Ratio %']}%")

# Identify best and worst months for profit ratio
max_ratio_month = monthly_sales.loc[monthly_sales['Profit Ratio %'].idxmax()]
min_ratio_month = monthly_sales.loc[monthly_sales['Profit Ratio %'].idxmin()]

print(f"\nHighest Profit Ratio Month: {max_ratio_month['Order Month Name']}")
print(f"  Profit Ratio: {max_ratio_month['Profit Ratio %']}%")
print(f"  Sales: ${max_ratio_month['Sales']:,.2f}")
print(f"  Profit: ${max_ratio_month['Profit']:,.2f}")

print(f"\nLowest Profit Ratio Month: {min_ratio_month['Order Month Name']}")
print(f"  Profit Ratio: {min_ratio_month['Profit Ratio %']}%")
print(f"  Sales: ${min_ratio_month['Sales']:,.2f}")
print(f"  Profit: ${min_ratio_month['Profit']:,.2f}")

# Visualization 5: Profit Ratio Analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Profit Ratio Analysis Across Dimensions', fontsize=16, fontweight='bold')

# Plot 1: Profit Ratio by Month
axes[0].plot(monthly_sales['Order Month Name'], monthly_sales['Profit Ratio %'], 
             marker='o', linewidth=2, markersize=8, color='blue')
axes[0].fill_between(monthly_sales['Order Month Name'], 
                     monthly_sales['Profit Ratio %'], alpha=0.3, color='lightblue')
axes[0].set_title('Monthly Profit Ratio Trend', fontweight='bold')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Profit Ratio %')
axes[0].tick_params(axis='x', rotation=45)
axes[0].axhline(y=monthly_sales['Profit Ratio %'].mean(), color='r', linestyle='--', 
                label=f"Avg: {monthly_sales['Profit Ratio %'].mean():.1f}%")
axes[0].legend()

# Plot 2: Profit Ratio by Category
colors_cat_ratio = ['green' if x >= 0 else 'red' for x in category_performance['Profit Ratio %']]
axes[1].bar(category_performance['Category'], category_performance['Profit Ratio %'], 
            color=colors_cat_ratio)
axes[1].set_title('Profit Ratio by Category', fontweight='bold')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Profit Ratio %')
axes[1].axhline(y=0, color='black', linewidth=0.8)
for i, v in enumerate(category_performance['Profit Ratio %']):
    axes[1].text(i, v, f'{v}%', ha='center', va='bottom' if v >= 0 else 'top', 
                 fontweight='bold', color='white' if abs(v) > 5 else 'black')

# Plot 3: Profit Ratio by Segment
colors_seg_ratio = ['green' if x >= 0 else 'red' for x in segment_performance['Profit Ratio %']]
axes[2].bar(segment_performance['Segment'], segment_performance['Profit Ratio %'], 
            color=colors_seg_ratio)
axes[2].set_title('Profit Ratio by Customer Segment', fontweight='bold')
axes[2].set_xlabel('Segment')
axes[2].set_ylabel('Profit Ratio %')
axes[2].axhline(y=0, color='black', linewidth=0.8)
for i, v in enumerate(segment_performance['Profit Ratio %']):
    axes[2].text(i, v, f'{v}%', ha='center', va='bottom' if v >= 0 else 'top', 
                 fontweight='bold', color='white' if abs(v) > 5 else 'black')

plt.tight_layout()
plt.show()

# ============================================
# 8. COMPREHENSIVE REPORT
# ============================================

print("\n" + "="*50)
print("8. COMPREHENSIVE ANALYSIS REPORT")
print("="*50)

print("\n" + "="*60)
print("EXECUTIVE SUMMARY")
print("="*60)

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   • Total Sales: ${df['Sales'].sum():,.2f}")
print(f"   • Total Profit: ${df['Profit'].sum():,.2f}")
print(f"   • Average Profit Margin: {overall_ratio}%")
print(f"   • Total Orders: {df.shape[0]}")
print(f"   • Time Period: {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")

print(f"\n🏆 TOP PERFORMERS:")
print(f"   • Highest Sales Month: {max_sales_month['Order Month Name']} (${max_sales_month['Sales']:,.0f})")
print(f"   • Highest Profit Month: {max_profit_month['Order Month Name']} (${max_profit_month['Profit']:,.0f})")
print(f"   • Best Category by Sales: {max_sales_category['Category']}")
print(f"   • Best Category by Profit: {max_profit_category['Category']}")

print(f"\n⚠️  AREAS NEEDING ATTENTION:")
print(f"   • Lowest Sales Month: {min_sales_month['Order Month Name']} (${min_sales_month['Sales']:,.0f})")
print(f"   • Lowest Profit Month: {min_profit_month['Order Month Name']} (${min_profit_month['Profit']:,.0f})")
print(f"   • Lowest Sales Category: {min_sales_category['Category']}")

print(f"\n👥 CUSTOMER INSIGHTS:")
for _, row in segment_performance.iterrows():
    print(f"   • {row['Segment']}: {row['Unique Customers']} customers, "
          f"${row['Avg Order Value']:.0f} avg order, {row['Profit Margin %']}% margin")

print(f"\n📈 KEY RECOMMENDATIONS:")
print("   1. Focus marketing efforts on low-performing months")
print("   2. Optimize product mix based on profit margin analysis")
print("   3. Develop targeted strategies for each customer segment")
print("   4. Monitor and improve underperforming sub-categories")
print("   5. Implement promotions during seasonal lows")

print(f"\n" + "="*60)
print("DETAILED PERFORMANCE METRICS")
print("="*60)

# Create summary DataFrame
summary_data = {
    'Metric': ['Total Sales', 'Total Profit', 'Avg Profit Margin', 'Total Orders',
               'Avg Order Value', 'Best Month (Sales)', 'Best Month (Profit)',
               'Worst Month (Sales)', 'Worst Month (Profit)'],
    'Value': [f"${df['Sales'].sum():,.2f}", 
              f"${df['Profit'].sum():,.2f}", 
              f"{overall_ratio}%",
              f"{df.shape[0]}",
              f"${(df['Sales'].sum()/df.shape[0]):.2f}",
              f"{max_sales_month['Order Month Name']} (${max_sales_month['Sales']:,.0f})",
              f"{max_profit_month['Order Month Name']} (${max_profit_month['Profit']:,.0f})",
              f"{min_sales_month['Order Month Name']} (${min_sales_month['Sales']:,.0f})",
              f"{min_profit_month['Order Month Name']} (${min_profit_month['Profit']:,.0f})"]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\n" + "="*60)
print("CATEGORY PERFORMANCE RANKING")
print("="*60)
print(category_performance[['Category', 'Sales', 'Profit', 'Profit Margin %', 'Avg Order Value']].to_string(index=False))

print(f"\n" + "="*60)
print("DATA QUALITY CHECK")
print("="*60)

# Check for data issues
print(f"Missing Values Check:")
print(df.isnull().sum())

print(f"\nNegative Profit Orders: {(df['Profit'] < 0).sum()} "
      f"({(df['Profit'] < 0).sum()/df.shape[0]*100:.1f}% of total)")

print(f"\nHigh Discount Orders (>30%): {(df['Discount'] > 0.3).sum()} "
      f"({(df['Discount'] > 0.3).sum()/df.shape[0]*100:.1f}% of total)")

# Save results to CSV files
print(f"\n💾 Saving analysis results to CSV files...")
monthly_sales.to_csv('monthly_sales_analysis.csv', index=False)
category_performance.to_csv('category_performance.csv', index=False)
subcategory_performance.to_csv('subcategory_performance.csv', index=False)
segment_performance.to_csv('segment_performance.csv', index=False)

print("✅ Analysis complete! Files saved:")
print("   - monthly_sales_analysis.csv")
print("   - category_performance.csv")
print("   - subcategory_performance.csv")
print("   - segment_performance.csv")
print(f"\n📊 Total visualizations generated: 5 comprehensive charts")