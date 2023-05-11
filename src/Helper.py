# Choose two groups from the Table 1

last_misery_agg = 'Last Misery'
most_pleasure_agg = 'Most Pleasure'
average = 'Average'

GRS_dataset = '../resources/GRS_dataset.xlsx'
title_sim = 'COSINE SIMILARITY between the rated movies and the movies to be recommended | Most similar to least similar order'

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
