import pandas as pd

from Helper import table_1, table_3_dict

last_misery_agg = 'Last Misery'
most_pleasure_agg = 'Most Pleasure'
average = 'Average'


def load_data():
    movies_vector_xlsx = pd.read_excel('../resources/GRS_dataset.xlsx', sheet_name='Movies Vector Representation')
    users_movie_ratings = pd.read_excel('../resources/GRS_dataset.xlsx', sheet_name='Users-Movie Ratings')

    rating_matrix_without_average = users_movie_ratings.iloc[0:50, 0:]
    rating_matrix_with_average = users_movie_ratings.iloc[54:, 0:]

    return movies_vector_xlsx, rating_matrix_without_average, rating_matrix_with_average


def prepare_dataframes(movies_vector_xlsx):
    movies_to_recommend = movies_vector_xlsx.iloc[0:15, 1:]
    movies_to_recommend_selected = movies_to_recommend.iloc[:, 0:2]
    movies_to_recommend_selected['Id Number'] = movies_to_recommend_selected['Id Number'].astype(int)
    movies_rated_by_user = movies_vector_xlsx.iloc[16:, 1:]

    return movies_to_recommend_selected, movies_rated_by_user


def load_similarity_data():
    similarity_Rated_Movies_newMov_xlsx = pd.read_excel('../resources/GRS_dataset.xlsx',
                                                        sheet_name='Similarity RatedMovies & NewMov')
    cos_similarity_between_moviesRat_and_moviesReco = similarity_Rated_Movies_newMov_xlsx.iloc[1:, :]

    return cos_similarity_between_moviesRat_and_moviesReco


def get_dataframes_dict(df):
    df = df.rename(columns={
        'COSINE SIMILARITY between the rated movies and the movies to be recommended | Most similar to least similar order': 'Unnamed:0'})
    keys_columns = [col for col_idx, col in enumerate(df.columns) if col_idx % 2 == 0]

    dataframes_dict = {}

    for key_col_idx, key_col in enumerate(keys_columns):
        key = df.iloc[0][key_col]

        ratings_col = df.columns[key_col_idx * 2 + 1]

        sub_df = df[[key_col, ratings_col]].drop(1)
        sub_df = sub_df.dropna()

        sub_df.columns = ['Title', 'Rating']

        dataframes_dict[key] = sub_df

    return dataframes_dict


def calculate_aggregations(users, user_movie_ratings):
    group_ratings = user_movie_ratings[users]
    group_ratings = group_ratings.T
    avg_ratings = group_ratings.mean(axis=0)
    min_ratings = group_ratings.min(axis=0)
    max_ratings = group_ratings.max(axis=0)

    return avg_ratings, min_ratings, max_ratings


def construct_matrix_with_aggregations(users, user_movie_ratings):
    avg_ratings, min_ratings, max_ratings = calculate_aggregations(users, user_movie_ratings)
    user_movie_ratings_avg = user_movie_ratings_lm = user_movie_ratings_mp = user_movie_ratings.copy()
    user_movie_ratings_avg[average] = avg_ratings
    user_movie_ratings_lm[last_misery_agg] = min_ratings
    user_movie_ratings_mp[most_pleasure_agg] = max_ratings

    return user_movie_ratings_avg, user_movie_ratings_lm, user_movie_ratings_mp


def get_most_similar_movies_dict(selected_columns_aggregation, dataframes_dict):
    most_similar_movies_dict = {}

    for index, row in selected_columns_aggregation.iterrows():
        movie_title = row.iloc[1]
        if type(movie_title) == str:
            movie_df = dataframes_dict[movie_title.strip('\n')]
        else:
            movie_df = dataframes_dict[movie_title]

        highest_rating = movie_df['Rating'].max()

        if highest_rating > 0:
            highest_rated_movies = movie_df[movie_df['Rating'] == highest_rating]
            most_similar_movies_dict[movie_title] = highest_rated_movies

    return most_similar_movies_dict


def get_rating_multiplication_dataframes(selected_columns_Last_Misery, most_similar_movies_dict):
    rating_multiplication_dataframes = []

    for index, row in selected_columns_Last_Misery.iterrows():
        movie_title = row.iloc[1]
        movie_rating = row.iloc[-1]
        similar_movies_df = most_similar_movies_dict.get(movie_title)

        if similar_movies_df is not None:
            similar_movies_df = similar_movies_df.copy()
            similar_movies_df['Ranked List Recommendation'] = similar_movies_df['Rating'] * movie_rating
            rating_multiplication_dataframes.append(similar_movies_df)

    return rating_multiplication_dataframes


def get_final_recommendations_dataframe(rating_multiplication_dataframes):
    resulting_dataframe = pd.concat(rating_multiplication_dataframes, ignore_index=True)
    max_ranked_list_recommendation_idx = resulting_dataframe.groupby('Title')['Ranked List Recommendation'].idxmax()
    unique_movies_with_highest_ranked_list_recommendation = resulting_dataframe.loc[
        max_ranked_list_recommendation_idx].reset_index(drop=True)
    unique_movies_with_highest_ranked_list_recommendation = unique_movies_with_highest_ranked_list_recommendation.sort_values(
        by='Ranked List Recommendation', ascending=False)

    return unique_movies_with_highest_ranked_list_recommendation


# Cargar los datos
movies_vector_xlsx, rating_matrix_without_average, rating_matrix_with_average = load_data()

# Preparar los DataFrames
movies_to_recommend_selected, movies_rated_by_user = prepare_dataframes(movies_vector_xlsx)

# Cargar datos de similitud
cos_similarity_between_moviesRat_and_moviesReco = load_similarity_data()

# Obtener diccionario de DataFrames
dataframes_dict = get_dataframes_dict(cos_similarity_between_moviesRat_and_moviesReco)


def format_recommendations(recommendations, group_id, strategy):
    top_3_movies = recommendations.head(3)
    movies_list = list(top_3_movies["Title"].values)

    # Completar la lista con '-' en caso de que tenga menos de 3 películas
    while len(movies_list) < 3:
        movies_list.append('-')

    result = pd.DataFrame(columns=["Group ID", "Strategy", "1st", "2nd", "3rd"])
    result.loc[0] = [group_id, strategy] + movies_list

    return result


def get_recommendations(user_movie_ratings_lm, dataframes_dict, name_aggregation, group_id):
    # Ordenar el DataFrame
    sorted_df = user_movie_ratings_lm.sort_values(by=name_aggregation, ascending=False)
    selected_columns_aggregation = sorted_df.iloc[:7, [0, 1, -1]]

    # Obtener el diccionario de películas más similares
    most_similar_movies_dict = get_most_similar_movies_dict(selected_columns_aggregation, dataframes_dict)

    # Obtener DataFrames de multiplicación de calificaciones
    rating_multiplication_dataframes = get_rating_multiplication_dataframes(selected_columns_aggregation,
                                                                            most_similar_movies_dict)

    # Obtener el DataFrame de recomendaciones finales
    final_recommendations = get_final_recommendations_dataframe(rating_multiplication_dataframes)

    formatted_recommendations = format_recommendations(final_recommendations, group_id, name_aggregation)

    return formatted_recommendations


def get_table_recommendations(user_movie_ratings_lm, user_movie_ratings_avg, user_movie_ratings_mp, dataframes_dict,
                              last_misery_agg, average, most_pleasure_agg, s_g):
    recommendations_lm = get_recommendations(user_movie_ratings_lm, dataframes_dict, last_misery_agg, s_g)
    recommendations_avg = get_recommendations(user_movie_ratings_avg, dataframes_dict, average, s_g)
    recommendations_mp = get_recommendations(user_movie_ratings_mp, dataframes_dict, most_pleasure_agg, s_g)

    result = pd.concat([recommendations_lm, recommendations_avg, recommendations_mp], ignore_index=True)
    return result


# Seleccionar dos grupos
# selected_groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
selected_groups = [1, 5]
# Eliminar las dos primeras columnas de rating_matrix_with_average
column_name = rating_matrix_with_average.columns[2]
user_movie_ratings = rating_matrix_with_average.drop(columns=column_name, axis=1)

# Crear un DataFrame vacío para almacenar las recomendaciones de todos los grupos
all_recommendations = pd.DataFrame()
for s_g in selected_groups:
    user_movie_ratings_avg, user_movie_ratings_lm, user_movie_ratings_mp = construct_matrix_with_aggregations(
        table_1[s_g], user_movie_ratings)

    # Utilizar la función get_recommendations() con user_movie_ratings como parámetro

    recommendations = get_table_recommendations(user_movie_ratings_lm, user_movie_ratings_avg, user_movie_ratings_mp,
                                                dataframes_dict,
                                                last_misery_agg, average, most_pleasure_agg, s_g)

    # Concatenar las recomendaciones del grupo actual al DataFrame de todas las recomendaciones
    all_recommendations = pd.concat([all_recommendations, recommendations], ignore_index=True)

print(all_recommendations)

estrategia_last_misery = all_recommendations[all_recommendations['Strategy'] == 'Last Misery']
estrategia_average = all_recommendations[all_recommendations['Strategy'] == 'Average']
estrategia_most_pleasure = all_recommendations[all_recommendations['Strategy'] == 'Most Pleasure']
estrategia_last_misery = estrategia_last_misery.drop(columns=['Strategy'])
estrategia_average = estrategia_average.drop(columns=['Strategy'])
estrategia_most_pleasure = estrategia_most_pleasure.drop(columns=['Strategy'])


def get_movie_titles_from_ids(movie_ids, movies_to_recommend_selected):
    movie_titles = []
    for movie_id in movie_ids:
        movie_title = \
        movies_to_recommend_selected.loc[movies_to_recommend_selected['Id Number'] == movie_id, 'Title'].iloc[0]
        movie_titles.append(movie_title)
    return movie_titles


table_3_titles_dict = {}
for group_id, movie_ids in table_3_dict.items():
    movie_titles = get_movie_titles_from_ids(movie_ids, movies_to_recommend_selected)
    table_3_titles_dict[group_id] = movie_titles

# conviértelo en un DataFrame
df = pd.DataFrame.from_dict(table_3_titles_dict, orient='index')

# opcionalmente, puedes establecer los nombres de las columnas
df.columns = ['1st', '2nd', '3rd']

# imprime el DataFrame
print(df)
