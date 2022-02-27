import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Data:
    """
    This class contains several methods for loading and preprocessing wine data.

    Methods
    -------
    load_data:
        Load CSV file
    clean_nan_strings:
        Removes data where the country or variety is null
        Replaces null strings in less-essential variables with 'None'
    clean_nan_values:
        Replaces nan prices with the mean for its country
    ohe_new_old_world:
        Manually one-hot-encodes the new world or old world status
    no_outliers:
        Returns a dataframe without price outliers (z < 3 default)
    outliers_only:
        Returns a dataframe of only price outliers (z >= 3 default)
    """
    def load_data(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        return df
    def clean_nan_strings(df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows where the country or variety is null
        df = df.dropna(subset=['country','variety'])
        # Replace null strings with 'None'
        replace_nan = ['designation',
                       'region_1',
                       'region_2',
                       'taster_name',
                       'taster_twitter_handle']
        for col in replace_nan:
            mask = df[col].isna()
            df.loc[mask, col] = 'None'
        return df
    def clean_nan_values(df: pd.DataFrame) -> pd.DataFrame:
        # Calculate the mean price for each country
        df_avg = df[['country','price']]
        df_avg = df_avg.groupby(by='country').mean()
        # Takes the average price from df_avg and replaces null prices in df
        # The .dropna() drops the one null value remaining
        df = df.set_index('country').combine_first(df_avg).reset_index().dropna()
        df = df.set_index('id')
        return df
    def ohe_new_old_world(df: pd.DataFrame) -> pd.DataFrame:
        # Apologies for the long, ugly lists!
        # New world vs. old world classifications from Wine Folly
        new_world_list = [
            'Argentina',
            'Australia',
            'Brazil',
            'Canada',
            'Chile',
            'China',
            'Egypt',
            'India',
            'Mexico',
            'New Zealand',
            'Peru',
            'South Africa',
            'US',
            'Uruguay'
        ]
        # Ancient world wines are grouped in with old world wines
        old_world_list = [
            'Armenia',
            'Austria',
            'Bosnia and Herzegovina',
            'Bulgaria',
            'Croatia',
            'Cyprus',
            'Czech Republic',
            'England',
            'France',
            'Georgia',
            'Germany',
            'Greece',
            'Hungary',
            'Israel',
            'Italy',
            'Lebanon',
            'Luxembourg',
            'Macedonia',
            'Moldova',
            'Morocco',
            'Portugal',
            'Romania',
            'Serbia',
            'Slovakia',
            'Slovenia',
            'Spain',
            'Switzerland',
            'Turkey',
            'Ukraine'
        ]
        # Creates boolean masks for new world and old world countries
        df['new_world'] = df['country'].apply(lambda x: x in new_world_list)
        df['old_world'] = df['country'].apply(lambda x: x in old_world_list)
        return df
    def no_outliers(df: pd.DataFrame, z=3) -> pd.DataFrame:
        # Remove price outliers
        df_no_outliers = df[(np.abs(stats.zscore(df['price'])) < z)]
        return df_no_outliers
    def outliers_only(df: pd.DataFrame, z=3) -> pd.DataFrame:
        # Keeps only price outliers
        df_outliers = df[(np.abs(stats.zscore(df['price'])) >= z)]
        return df_outliers


class Review:
    """
    This class contains several methods for processing text data.

    Methods
    -------
    lower:
        Returns the string in lowercase
    punct:
        Removes punctuation, including em-dashes
    remove_stopwords:
        Removes standard English stopwords
    remove_wine_stopwords:
        Removes standard English stopwords, plus wine-specific stopwords
    vector_substract:
        Calculates the difference of word vectors
    embed_sentence:
        Converts sentence to a matrix
    embed_pad:
        Converts a list of sentences into a list of matrices and pads them
    """
    def lower(rev: str) -> str:
        if type(rev)==str:
            return rev.lower()
        else:
            return rev.str.lower()

    def punct(rev: str) -> str:
        punc = string.punctuation
        punc += '—'
        for p in punc:
            rev = rev.replace(p, ' ')
        return rev

    def remove_stopwords(rev: str) -> str:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        stop_words = set(stopwords.words('english'))
        tokenized = word_tokenize(rev)
        without_stopwords = [
            word for word in tokenized if not word in stop_words
        ]
        return without_stopwords

    def remove_wine_stopwords(rev: str) -> str:
        wine_stopwords = stopwords.words('english')
        custom_words = ['wine', 'flavors', 'aromas', 'notes', 'palate']
        wine_stopwords.extend(custom_words)
        tokenized = word_tokenize(rev)
        without_stopwords = [
            word for word in tokenized if not word in wine_stopwords
        ]
        return without_stopwords

    def vector_subtract(wv_model, word1: str, word2: str) -> list:
        v_word1 = wv_model.wv[word1]
        v_word2 = wv_model.wv[word2]
        v_result = v_word1 - v_word2
        return wv_model.wv.similar_by_vector(v_result)

    def embed_review(wv_model, review):
        embed = []
        for word in review:
            if word in wv_model.wv:
                embed.append(wv_model.wv[word])
        embed = np.array(embed)
        if embed.shape[0] > 78:
            sentences_pad = pad_sequences(embed,
                                          dtype='float32',
                                          truncating='post')
        else:
            to_add = 78 - embed.shape[0]
            sentences_pad = np.pad(embed, pad_width=[(to_add,0),(0,0)])
        return np.expand_dims(sentences_pad, axis=0)
