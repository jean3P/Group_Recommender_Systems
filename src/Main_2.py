import pandas as pd

from Helper import table_1, table_3_dict, average, last_misery_agg, most_pleasure_agg, GRS_dataset, title_sim


# Función para cargar los datos del archivo excel
def load_data():
    # Cargar las dos hojas del archivo excel en dataframes de pandas
    movies_vector_xlsx = pd.read_excel(GRS_dataset, sheet_name='Movies Vector Representation')
    users_movie_ratings = pd.read_excel(GRS_dataset, sheet_name='Users-Movie Ratings')

    # Dividir el dataframe de calificaciones de los usuarios en dos partes: sin y con la fila de promedios
    rating_matrix_without_average = users_movie_ratings.iloc[0:50, 0:]
    rating_matrix_with_average = users_movie_ratings.iloc[54:, 0:]

    return movies_vector_xlsx, rating_matrix_without_average, rating_matrix_with_average


# Función para preparar los dataframes que se utilizarán en el análisis
def prepare_dataframes(movies_vector_xlsx):
    # Separar el dataframe de vectores de películas en dos partes:
    # las películas a recomendar y las películas calificadas por el usuario
    movies_to_recommend = movies_vector_xlsx.iloc[0:15, 1:]
    movies_to_recommend_selected = movies_to_recommend.iloc[:, 0:2]
    movies_to_recommend_selected['Id Number'] = movies_to_recommend_selected['Id Number'].astype(int)
    movies_rated_by_user = movies_vector_xlsx.iloc[16:, 1:]

    return movies_to_recommend_selected, movies_rated_by_user


# Función para cargar los datos de similitud del archivo excel
def load_similarity_data():
    # Cargar la hoja de similitud del archivo excel en un dataframe de pandas
    similarity_Rated_Movies_newMov_xlsx = pd.read_excel(GRS_dataset,
                                                        sheet_name='Similarity RatedMovies & NewMov')

    # Eliminar la primera fila del dataframe
    cos_similarity_between_moviesRat_and_moviesReco = similarity_Rated_Movies_newMov_xlsx.iloc[1:, :]

    return cos_similarity_between_moviesRat_and_moviesReco


def get_dataframes_dict_of_similarities(df):
    # Renombrar la columna del título de la película en el dataframe para un manejo más sencillo
    df = df.rename(columns={title_sim: 'Unnamed:0'})

    # Crear una lista con los nombres de las columnas con índice par en el dataframe
    keys_columns = [col for col_idx, col in enumerate(df.columns) if col_idx % 2 == 0]

    # Inicializar un diccionario vacío que se llenará con los sub-dataframes
    dataframes_dict = {}

    # Iterar sobre las columnas de las claves
    for key_col_idx, key_col in enumerate(keys_columns):
        # Extraer el nombre de la película como clave del primer valor en la columna
        key = df.iloc[0][key_col]

        # Identificar la columna de calificaciones que corresponde a la columna de claves actual
        ratings_col = df.columns[key_col_idx * 2 + 1]

        # Crear un sub-dataframe con solo la columna de claves y la columna de calificaciones, y eliminar la primera fila
        sub_df = df[[key_col, ratings_col]].drop(1)

        # Eliminar cualquier fila con un valor nulo
        sub_df = sub_df.dropna()

        # Renombrar las columnas del sub-dataframe para un manejo más sencillo
        sub_df.columns = ['Title', 'Rating']

        # Agregar el sub-dataframe al diccionario con la clave del título de la película
        dataframes_dict[key] = sub_df

    # Devolver el diccionario de sub-dataframes
    return dataframes_dict


def calculate_aggregations(users, user_movie_ratings):
    # Seleccionar las calificaciones de las películas para el grupo de usuarios dado
    group_ratings = user_movie_ratings[users]

    # Transponer el dataframe para que los usuarios sean las columnas y las películas sean las filas
    group_ratings = group_ratings.T

    # Calcular la calificación promedio para cada película
    avg_ratings = group_ratings.mean(axis=0)

    # Calcular la calificación mínima para cada película (Last misery)
    min_ratings = group_ratings.min(axis=0)

    # Calcular la calificación máxima para cada película (Most pleasure)
    max_ratings = group_ratings.max(axis=0)

    # Devolver las calificaciones promedio, mínimas y máximas
    return avg_ratings, min_ratings, max_ratings


def construct_matrix_with_aggregations(users, user_movie_ratings):
    # Calcular las calificaciones promedio, mínimas y máximas para el grupo de usuarios dado
    avg_ratings, min_ratings, max_ratings = calculate_aggregations(users, user_movie_ratings)

    # Crear copias del dataframe original de calificaciones de películas para cada estrategia de agregación
    user_movie_ratings_avg = user_movie_ratings_lm = user_movie_ratings_mp = user_movie_ratings.copy()

    # Agregar las calificaciones promedio al dataframe de calificaciones promedio
    user_movie_ratings_avg[average] = avg_ratings

    # Agregar las calificaciones mínimas al dataframe de calificaciones mínimas
    user_movie_ratings_lm[last_misery_agg] = min_ratings

    # Agregar las calificaciones máximas al dataframe de calificaciones máximas
    user_movie_ratings_mp[most_pleasure_agg] = max_ratings

    # Devolver los dataframes de calificaciones para cada estrategia de agregación
    return user_movie_ratings_avg, user_movie_ratings_lm, user_movie_ratings_mp


def get_most_similar_movies_dict(selected_columns_aggregation, dataframes_dict):
    # Crear un diccionario para almacenar las películas más similares
    most_similar_movies_dict = {}

    # Iterar sobre cada fila en la agregación seleccionada
    for index, row in selected_columns_aggregation.iterrows():
        # Obtener el título de la película
        movie_title = row.iloc[1]

        # Gestionar los títulos de las películas que son cadenas o no
        if type(movie_title) == str:
            movie_df = dataframes_dict[movie_title.strip('\n')]
        else:
            movie_df = dataframes_dict[movie_title]

        # Obtener la calificación más alta en el dataframe de la película
        highest_rating = movie_df['Rating'].max()

        # Si la calificación más alta es mayor que 0, añadir las películas con la calificación más alta al diccionario
        if highest_rating > 0:
            highest_rated_movies = movie_df[movie_df['Rating'] == highest_rating]
            most_similar_movies_dict[movie_title] = highest_rated_movies

    # Devolver el diccionario de películas más similares
    return most_similar_movies_dict


def get_rating_multiplication_dataframes(selected_columns_Last_Misery, most_similar_movies_dict):
    # Crear una lista para almacenar los dataframes de multiplicación de calificaciones
    rating_multiplication_dataframes = []

    # Iterar sobre cada fila en las columnas seleccionadas de la última miseria
    for index, row in selected_columns_Last_Misery.iterrows():
        # Obtener el título de la película y la calificación
        movie_title = row.iloc[1]
        movie_rating = row.iloc[-1]

        # Obtener el dataframe de películas similares para el título de la película
        similar_movies_df = most_similar_movies_dict.get(movie_title)

        # Si se encuentra un dataframe de películas similares, multiplicar las calificaciones
        # por la calificación de la película
        if similar_movies_df is not None:
            similar_movies_df = similar_movies_df.copy()
            similar_movies_df['Ranked List Recommendation'] = similar_movies_df['Rating'] * movie_rating
            rating_multiplication_dataframes.append(similar_movies_df)

    # Devolver la lista de dataframes de multiplicación de calificaciones
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
dataframes_dict = get_dataframes_dict_of_similarities(cos_similarity_between_moviesRat_and_moviesReco)


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
selected_groups = [1, 3]
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
