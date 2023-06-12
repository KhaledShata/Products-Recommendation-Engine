import re
import nltk
import nltk_download

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import streamlit as st
from PIL import Image

@st.cache_data
def load_txtembd():
    file_path = 'text_embeddings.pkl'  # Path to the pickle file
    with open(file_path, 'rb') as file:
        text_embeddings = pickle.load(file)
    return text_embeddings


@st.cache_data
def load_df():
    file_path = 'processed_df.zip'  # Path to the pickle file
    processed_df= pd.read_csv(file_path, low_memory=False, encoding='utf-8')
    return processed_df



@st.cache_data
def load_vectorizer():
    file_path = 'tfidf_vectorizer.pkl'  # Path to the pickle file
    with open(file_path, 'rb') as file:
        tfidf_vectorizer = pickle.load(file)

    return tfidf_vectorizer


text_embeddings = load_txtembd()
processed_df = load_df()
tfidf_vectorizer = load_vectorizer()




def preprocess_text(text):
    if pd.isnull(text):  # Check if text is null
        return ''
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text).lower())  # Convert to string before applying lower() function
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    processed_text = " ".join(filtered_tokens)
    return processed_text



def recommend_products(query, cat, store, sort_by_price=None, top_n=28):
    # Preprocess the query
    query = preprocess_text(query)

    # Transform the query into an embedding using the TF-IDF vectorizer
    query_embedding = tfidf_vectorizer.transform([query])

    # Calculate the cosine similarity between the query embedding and all product embeddings
    similarity_scores = cosine_similarity(query_embedding, text_embeddings)

    # Get the indices of the top-N most similar products
    top_indices = similarity_scores.argsort()[0][-1000:][::-1]

    # Retrieve the top-N recommended products
    recommendations = processed_df.iloc[top_indices]
    if store:
        recommendations = recommendations[recommendations['STORE_NAME'] == store]
    if cat:
        recommendations = recommendations[recommendations['CATEGORY'] == cat]

    recommendations = recommendations[:top_n]

    if sort_by_price == 'Low to High':
        recommendations = recommendations.sort_values('PRODUCT_PRICE')
    elif sort_by_price == 'High to Low':
        recommendations = recommendations.sort_values('PRODUCT_PRICE', ascending=False)


    recommendations_dic = []

    default_image_path = "https://nayemdevs.com/wp-content/uploads/2020/03/default-product-image.png"

    for index, row in recommendations.iterrows():
        product_dic = {}
        product_dic['product_image'] = row['IMAGE_URL']
        product_dic['product_name'] = row['PRODUCT_NAME']
        product_dic['product_brand'] = row['PRODUCT_BRAND']
        product_dic['product_price'] = row['PRODUCT_PRICE']
        product_dic['product_url'] = row['PRODUCT_LINK']
        product_dic['product_store'] = row['STORE_NAME']

        if pd.isnull(product_dic['product_image']):
            product_dic['product_image'] = default_image_path
        # Resize the image to 498 x 679
        # try:
        #     response = requests.get(product_dic['product_image'], stream=True)
        #     response.raise_for_status()
        #     image = Image.open(response.raw)
        #     if image.mode == 'P':
        #         image = image.convert('RGB')
        #     image = image.resize((498, 679))
        #     buffered = BytesIO()
        #     image.save(buffered, format="JPEG")
        #     product_dic['product_image'] = buffered.getvalue()
        # except:
        #     response = requests.get(default_image_path, stream=True)
        #     response.raise_for_status()
        #     image = Image.open(response.raw)
        #     if image.mode == 'P':
        #         image = image.convert('RGB')
        #     image = image.resize((498, 679))
        #     buffered = BytesIO()
        #     image.save(buffered, format="JPEG")
        #     product_dic['product_image'] = buffered.getvalue()


        recommendations_dic.append(product_dic)
    return recommendations_dic

# PAGE STYLING
st.markdown(f"""
    <style>
        p {{
            margin-bottom: 0;
        }}
        div[data-testid="column"]:nth-of-type(n+3) {{
            margin-bottom: 1rem;
        }}
        .block-container {{
            padding: 0
        }}
        footer {{
            margin-top: 100px
            display: none
        }}
        .stButton {{
            margin-top: .5rem
        }}
    </style>""",
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    .product-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        position: relative;
    }

    .product-row:after {
        content: "";
        position: absolute;
        width: 100%;
        height: 3px;
        background-color: black;
        bottom: 0;
        left: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def show_page():
    n_cols = 4
    k = 0

    st.markdown('<h1 class="title">Products Recommendation System</h1>', unsafe_allow_html=True)
    text_input = st.text_input("Enter your search query:", key="search_input")
    cat = st.selectbox("Filter By Category :",
                       [None, 'Food & Beverage', 'Other', 'Special Occasions & Gifts',
                        'Baby & Infant', 'Health & Personal Care', 'Home & Garden',
                        'Books & Magazines', 'Arts, Crafts & School Supplies',
                        'Clothing & Accessories', 'Electronics & Accessories',
                        'Sports & Outdoors', 'Pets', 'Household & Cleaning'])

    store = st.selectbox("Filter By Store :",
                         [None, 'Amazon', 'Oda', 'REWE', 'Wolt: Flink Karl Liebknecht', 'Flink',
                          'Mitte Meer Charlottenburg', 'Muller', 'Asia Food Tuan Lan',
                          'EDEKA', 'PENNY', 'Asia24 GmbH', 'Wolt: Latino Point',
                          'Amore Store', 'FrischeParadies', 'Goldhahn & sampson',
                          'Wolt: Miconbini', "Golly's", 'ASIA4FRIENDS',
                          'Original Unverpackt', 'Veganz', 'FLINK', 'Tante Emma',
                          'Mitte Meer'])

    sort_by_price = st.selectbox("Sort by Price:", ['Relevance', 'Low to High', 'High to Low'])

    search_button = st.button("Search")

    if search_button or st.session_state.search_input:

        recommendations = recommend_products(text_input, cat, store, sort_by_price=sort_by_price)

        st.markdown('<h3>Recommended Products:</h3>', unsafe_allow_html=True)



        for i, recommendation in enumerate(recommendations):
            for j in ["product_image", "product_name", "product_brand", "product_price"]:
                if pd.isnull(recommendation[j]):
                    recommendation[j] = "N/A"

            if k % n_cols == 0:
                if k > 0:
                    st.markdown('<div class="product-row">', unsafe_allow_html=True)
                    st.markdown('', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                cols = st.columns(n_cols)

            with st.container():

                with cols[k % n_cols]:
                    st.image(recommendation['product_image'], use_column_width=True)

                    st.markdown('[' + recommendation['product_name'] + '](' + recommendation['product_url'] + ')',
                                unsafe_allow_html=True)
                    st.write("**" + str(recommendation['product_store']) + "**")
                    st.markdown(
                        "<p style='font-size: 20px'><b><font color='red'>" + str(recommendation['product_price']) + " €</font></b></p>",
                        unsafe_allow_html=True)

            k += 1

show_page()


