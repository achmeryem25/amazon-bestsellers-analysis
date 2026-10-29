import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. LOAD THE DATA ----------
df = pd.read_csv("bestsellers with categories.csv")
print(df.head())

# ---------- 2. EXPLORE ----------
print("\nShape (rows, columns):", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nSummary statistics:")
print(df.describe())

# ---------- 3. CLEAN ----------
# Rename columns so they are easier to type
df.columns = ["name", "author", "rating", "reviews", "price", "year", "genre"]

# The same book appears once per year it was a bestseller
print("\nTotal rows:", len(df))
print("Unique book titles:", df["name"].nunique())

# Keep one row per title for "unique book" analysis
unique_books = df.drop_duplicates(subset="name")

# ---------- 4. ANALYSIS ----------
# Fiction vs Non Fiction split (unique books)
print("\nBooks per genre:")
print(unique_books["genre"].value_counts())

# Top 10 authors by number of unique bestsellers
print("\nTop 10 authors:")
print(unique_books["author"].value_counts().head(10))

# Top 10 most reviewed books
print("\nTop 10 most reviewed books:")
top_reviews = unique_books.sort_values("reviews", ascending=False).head(10)
print(top_reviews[["name", "author", "reviews"]])

# Highest rated books
print("\nHighest rated books:")
top_rated = unique_books.sort_values("rating", ascending=False).head(10)
print(top_rated[["name", "author", "rating"]])

# Average price, rating and reviews by genre
print("\nAverages by genre:")
print(df.groupby("genre")[["price", "rating", "reviews"]].mean().round(2))

# Average price per year
price_by_year = df.groupby("year")["price"].mean().round(2)
print("\nAverage price per year:")
print(price_by_year)

# Correlation between numeric columns
print("\nCorrelation:")
print(df[["rating", "reviews", "price", "year"]].corr().round(2))

# ---------- 5. VISUALIZE ----------
# Chart 1: books per genre
unique_books["genre"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Bestsellers by Genre")
plt.ylabel("Number of books")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("genre_counts.png")
plt.show()

# Chart 2: average price per year
price_by_year.plot(kind="line", marker="o")
plt.title("Average Price of Bestsellers per Year")
plt.xlabel("Year")
plt.ylabel("Price ($)")
plt.tight_layout()
plt.savefig("price_by_year.png")
plt.show()

# Chart 3: top 10 authors
unique_books["author"].value_counts().head(10).plot(kind="barh")
plt.title("Top 10 Authors by Number of Bestsellers")
plt.xlabel("Number of books")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_authors.png")
plt.show()

# Chart 4: rating distribution
df["rating"].plot(kind="hist", bins=15, edgecolor="black")
plt.title("Distribution of User Ratings")
plt.xlabel("Rating")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()

# ---------- 6. SAVE RESULTS ----------
unique_books.to_csv("unique_bestsellers.csv", index=False)
print("\nDone. Charts and cleaned CSV saved.")