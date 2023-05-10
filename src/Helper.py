# import pandas as pd
#
# # Load the data
# movies_vector_xlsx = pd.read_excel('../resources/GRS_dataset.xlsx', sheet_name='Movies Vector Representation')
# users_movie_ratings = pd.read_excel('../resources/GRS_dataset.xlsx', sheet_name='Users-Movie Ratings')
#
# # Eliminar la primera fila y columna
# rating_matrix_without_average = users_movie_ratings.iloc[0:50, 0:]
# rating_matrix_with_average = users_movie_ratings.iloc[54:, 0:]
#
# movies_to_recommend = movies_vector_xlsx.iloc[0:15, 1:]
# movies_to_recommend_selected = movies_to_recommend.iloc[:, 0:2]
# movies_to_recommend_selected['Id Number'] = movies_to_recommend_selected['Id Number'].astype(int)
# movies_rated_by_user = movies_vector_xlsx.iloc[16:, 1:]
#
# similarity_Rated_Movies_newMov_xlsx = pd.read_excel('../resources/GRS_dataset.xlsx', sheet_name='Similarity RatedMovies & NewMov')
#
# cos_similarity_between_moviesRat_and_moviesReco = similarity_Rated_Movies_newMov_xlsx.iloc[1:, :]
# # Asumiendo que tu DataFrame es cos_similarity_between_moviesRat_and_moviesReco
# df = cos_similarity_between_moviesRat_and_moviesReco
#
# # Filtra las columnas que contienen las keys (títulos)
# df = df.rename(columns={'COSINE SIMILARITY between the rated movies and the movies to be recommended | Most similar to least similar order': 'Unnamed:0'})
# keys_columns = [col for col_idx, col in enumerate(df.columns) if col_idx % 2 == 0]
#
# # Crea un diccionario vacío para almacenar los DataFrames
# dataframes_dict = {}
#
# # Itera sobre las columnas de keys y crea un DataFrame para cada key
# for key_col_idx, key_col in enumerate(keys_columns):
#     # Obtiene la key (título)
#     key = df.iloc[0][key_col]
#
#     # Encuentra la columna de ratings correspondiente a la key
#     ratings_col = df.columns[key_col_idx * 2 + 1]
#
#     # Elimina la primera fila (la que contiene las keys) y la columna de NaNs
#     sub_df = df[[key_col, ratings_col]].drop(1)
#     sub_df = sub_df.dropna()
#
#     # Renombra las columnas para que sean consistentes en todos los DataFrames
#     sub_df.columns = ['Title', 'Rating']
#
#     # Añade el DataFrame al diccionario utilizando la key
#     dataframes_dict[key] = sub_df

# print(rating_matrix_with_average)

# Choose two groups from the Table 1
table_1 = {
    1: ['U10', 'U26', 'U30', 'U12', 'U11', 'U16', 'U37', 'U29', 'U36'],
    2: ['U2', 'U3', 'U28', 'U44', 'U41'],
    3: ['U11', 'U15', 'U29'],
    4: ['U22', 'U50', 'U48'],
    5: ['U31', 'U32', 'U33', 'U34', 'U40', 'U42', 'U18', 'U21', 'U48'],
    6: ['U7', 'U8', 'U24', 'U25', 'U44'],
    7: ['U1', 'U6', 'U29'],
    8: ['U14', 'U27', 'U48'],
    9: ['U23', 'U10', 'U16', 'U36', 'U48'],
    10: ['U4', 'U20', 'U28', 'U47', 'U46', 'U45', 'U39', 'U49', 'U19'],
    11: ['U17', 'U2', 'U43', 'U5', 'U47'],
    12: ['U20', 'U28', 'U48', 'U35', 'U4'],
    13: ['U1', 'U51', 'U52', 'U53', 'U54', 'U55', 'U56', 'U57', 'U58'],
    14: ['U4', 'U19', 'U45', 'U46', 'U49'],
    15: ['U39', 'U45', 'U47']
}

table_3_dict = {
    1: [3, 2, 13],
    2: [11, 3, 4],
    3: [2, 3, 11],
    4: [6, 3, 15],
    5: [15, 3, 9],
    6: [11, 3, 2],
    7: [13, 5, 8],
    8: [6, 15, 2],
    9: [9, 3, 2],
    10: [5, 3, 13],
    11: [5, 1, 2],
    12: [3, 5, 6],
    13: [1, 8, 11],
    14: [6, 5, 14],
    15: [13, 5, 6]
}


#
# def calculate_aggregations(users, user_movie_ratings):
#     group_ratings = user_movie_ratings[users]
#     group_ratings = group_ratings.T
#     # Apply aggregation strategies
#     avg_ratings = group_ratings.mean(axis=0)
#     min_ratings = group_ratings.min(axis=0)
#     max_ratings = group_ratings.max(axis=0)
#
#     return avg_ratings, min_ratings, max_ratings
#
#
# def construct_matrix_with_aggregations(users, user_movie_ratings):
#     avg_ratings, min_ratings, max_ratings = calculate_aggregations(users, user_movie_ratings)
#     # user_movie_ratings[f'Average'] = avg_ratings
#     user_movie_ratings[f'Last Misery'] = min_ratings
#     # user_movie_ratings[f'Most Pleasure'] = max_ratings
#
#     return user_movie_ratings
#
# # Select two groups
# selected_groups = [2]
#
# # Eliminar las dos primeras columnas de rating_matrix_with_average
# column_name = rating_matrix_with_average.columns[2]
# user_movie_ratings = rating_matrix_with_average.drop(columns=column_name, axis=1)
#
# for s_g in selected_groups:
#     final = construct_matrix_with_aggregations(table_1[s_g], user_movie_ratings)
#
# # Asumiendo que tu DataFrame es df
# sorted_df = final.sort_values(by='Last Misery', ascending=False)
# selected_columns_Last_Misery = sorted_df.iloc[:7, [0, 1, -1]]
#
# # Crea una lista vacía para almacenar las películas más similares
# # Crea un diccionario vacío para almacenar las películas más similares
# most_similar_movies_dict = {}
#
# # Itera sobre las filas del DataFrame selected_columns_Last_Misery
# for index, row in selected_columns_Last_Misery.iterrows():
#     # Obtiene el título de la película de la columna 2
#     movie_title = row.iloc[1]
#
#     # Encuentra el DataFrame correspondiente en dataframes_dict utilizando la key (título de la película)
#     movie_df = dataframes_dict[movie_title]
#
#     # Encuentra el rating más alto
#     highest_rating = movie_df['Rating'].max()
#
#     # Si el rating más alto es mayor que 0, encuentra todas las películas con ese rating
#     if highest_rating > 0:
#         highest_rated_movies = movie_df[movie_df['Rating'] == highest_rating]
#
#         # Añade el DataFrame de películas más similares al diccionario usando el título de la película como clave
#         most_similar_movies_dict[movie_title] = highest_rated_movies
#
#
# # Convierte la lista de películas más similares en un DataFrame
# # most_similar_movies_df = pd.DataFrame(most_similar_movies, columns=['Most Similar Movie', 'Rating'])
# # Añade las columnas del DataFrame most_similar_movies_df a selected_columns_df
# selected_columns_Last_Misery.reset_index(drop=True, inplace=True)
#
# rating_multiplication_dataframes = []
#
# # Itera sobre las filas del DataFrame selected_columns_Last_Misery
# for index, row in selected_columns_Last_Misery.iterrows():
#     # Obtiene el título de la película de la columna 2
#     movie_title = row.iloc[1]
#
#     # Obtiene el rating de la película
#     movie_rating = row.iloc[-1]
#
#     # Encuentra el DataFrame de películas más similares en el diccionario utilizando el título de la película como clave
#     similar_movies_df = most_similar_movies_dict.get(movie_title)
#
#     # Si el DataFrame de películas más similares existe, multiplica el rating de la película por el rating de cada película similar
#     if similar_movies_df is not None:
#         # Crea una copia del DataFrame antes de modificarlo
#         similar_movies_df = similar_movies_df.copy()
#         similar_movies_df['Ranked List Recommendation'] = similar_movies_df['Rating'] * movie_rating
#
#         # Añade el DataFrame con los resultados de la multiplicación a la lista
#         rating_multiplication_dataframes.append(similar_movies_df)
#
# # Combina todos los DataFrames de la lista en un solo DataFrame
# resulting_dataframe = pd.concat(rating_multiplication_dataframes, ignore_index=True)
#
# # Agrupa el DataFrame por la columna "Title" y encuentra el índice con el valor más alto en la columna "Ranked List Recommendation"
# max_ranked_list_recommendation_idx = resulting_dataframe.groupby('Title')['Ranked List Recommendation'].idxmax()
#
# # Filtra el DataFrame utilizando los índices encontrados y crea un nuevo DataFrame sin títulos duplicados
# unique_movies_with_highest_ranked_list_recommendation = resulting_dataframe.loc[max_ranked_list_recommendation_idx].reset_index(drop=True)
# unique_movies_with_highest_ranked_list_recommendation = unique_movies_with_highest_ranked_list_recommendation.sort_values(by='Ranked List Recommendation', ascending=False)