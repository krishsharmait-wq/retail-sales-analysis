import pandas as pd
import sqlite3
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='pipeline.log', level=logging.INFO,
                     format='%(asctime)s - %(message)s')

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)
        logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def handle_missing_postal_code(self):
        self.df = self.df.dropna(subset=['Postal Code'])
        logging.info(f"Missing Postal Code rows dropped. New shape: {self.df.shape}")
        return self.df

    def remove_sales_outliers(self):
        Q1 = self.df['Sales'].quantile(0.25)
        Q3 = self.df['Sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        self.df = self.df[(self.df['Sales'] >= lower_bound) & (self.df['Sales'] <= upper_bound)]
        logging.info(f"Outliers removed from Sales. New shape: {self.df.shape}")
        return self.df

    def encode_categorical(self):
        self.df = pd.get_dummies(self.df, columns=['Category', 'Sub-Category', 'Region', 'Segment'])
        logging.info(f"Categorical columns encoded. New shape: {self.df.shape}")
        return self.df

class StorageHandler:
    def __init__(self, df, db_name='retail.db'):
        self.df = df
        self.db_name = db_name

    def save_to_db(self, table_name='sales_data'):
        conn = sqlite3.connect(self.db_name)
        self.df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        logging.info(f"Data saved to {self.db_name}, table: {table_name}")

class Analyzer:
    def __init__(self, df):
        self.df = df

    def sales_by_region(self):
        cols = [c for c in self.df.columns if c.startswith('Region_')]
        region_sales = {}
        for c in cols:
            region_sales[c] = self.df.loc[self.df[c] == True, 'Sales'].sum()
        return region_sales

    def top_products(self, n=10):
        return self.df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(n)

    def monthly_trend(self):
        temp = self.df.copy()
        temp['Order Date'] = pd.to_datetime(temp['Order Date'], dayfirst=True, errors='coerce')
        temp['Month'] = temp['Order Date'].dt.to_period('M')
        return temp.groupby('Month')['Sales'].sum()

    def generate_charts(self):
        region_data = self.sales_by_region()
        plt.figure(figsize=(8,5))
        plt.bar(region_data.keys(), region_data.values(), color='steelblue')
        plt.title('Region-wise Sales')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig('region_sales.png')
        plt.close()

        monthly_data = self.monthly_trend()
        plt.figure(figsize=(10,5))
        monthly_data.plot(kind='line', marker='o', color='darkorange')
        plt.title('Monthly Sales Trend')
        plt.ylabel('Total Sales')
        plt.xlabel('Month')
        plt.tight_layout()
        plt.savefig('monthly_trend.png')
        plt.close()

        logging.info("Charts saved: region_sales.png, monthly_trend.png")

if __name__ == "__main__":
    loader = DataLoader("train.csv")
    df = loader.load_data()

    cleaner = DataCleaner(df)
    df = cleaner.handle_missing_postal_code()
    df = cleaner.remove_sales_outliers()
    df = cleaner.encode_categorical()

    storage = StorageHandler(df)
    storage.save_to_db()

    analyzer = Analyzer(df)
    print("\n--- Region-wise Sales ---")
    print(analyzer.sales_by_region())

    print("\n--- Top 10 Products ---")
    print(analyzer.top_products())

    print("\n--- Monthly Sales Trend ---")
    print(analyzer.monthly_trend())

    analyzer.generate_charts()

    print("\nPipeline completed. Check retail.db, pipeline.log, and chart PNGs")