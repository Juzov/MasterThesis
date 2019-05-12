import random
import math
import requests
import json
import csv

def half_norm_bins(songs):
    """ create bins based on distance
    With the half normal distribution
    68.. percent in the first bin
    95-68.. percent in the second bin
    rest in the third bin
    """
    distances = {i: x['distance'] for i, x in enumerate(songs)}
    std = 0
    count = 0
    for x in distances.values():
        # mean is zero
        std += x ** 2
        count += 1

    std = math.sqrt(std / float(count - 1))
    bins = {0: [], 1: [], 2: []}
    for k, v in distances.items():
        if(v <= std):
            bins[0].append(songs[k])
        elif(v <= std * 2):
            bins[1].append(songs[k])
        else:
            bins[2].append(songs[k])
    return bins


def sample_bins(bins, sample_size=10):
    """ Sample from our created bins
    force a probability on the sample
    """
    for _ in range(0, sample_size):
        k = min(len(bins[0]), int(0.6827 * sample_size))
        r_0 = random.sample(bins[0], k=k)
        k = min(len(bins[1]), int((0.9545 - 0.6827) * sample_size))
        r_1 = random.sample(bins[1], k=k)
        k = min(len(bins[2]), int((1 - 0.9545) * sample_size))
        r_2 = random.sample(bins[2], k=k)
    return sort_sample(r_0) + sort_sample(r_1) + sort_sample(r_2)


def sort_sample(sample):
    return sorted(sample, key=lambda x: x['distance'])


def add_playable_links(songs):
    """ Fetch preview link
    Using 'track'
    Append 'link' if exists
    otherwise discard song
    """

    tracks = [song['track'] for song in songs]
    # tracks = {'tracks': tracks}
    ext_url = 'https://radio.api.soundtrackyourbrand.com/tracks/multi'
    headers = {'X-API-Version': '11',
               'Content-Type': 'application/json',
               'Authorization': 'Bearer eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE1NTc5MDc3NDAsImlhdCI6MTU1NjU0ODgyMywic3ViIjoiVlhObGNpd3NNV2syTkRBemNYWjRPR2N2IiwidHlwIjoic3VwZXJ1c2VyIn0.6r8GhgQDdb9T7jd9rzn_wCSO97XN2pZqxL7IhJtxkmmn4d8z2X-EsyUOLawBnpSbCeAoNvFlaSdUF7NJfFuifw'
               }

    get_request = requests.post(
        url=ext_url, data=json.dumps(tracks), headers=headers)
    # print(get_request)
    get_json = get_request.json()
    print

    # make sure every track is mapped to the right song.
    # Discard those who do not contain a preview link
    # append link otherwise
    # Note: Didn't want to divide into map and filter
    # instead creates a new dict

    songs = [{**song, **{'link': get_json[song['track']]['external_keys']['spotify']['preview_url']}}
             for song in songs
             if get_json[song['track']]['external_keys']['spotify']['preview_url']]

    return songs


def append_to_csv(get, cluster_index, data, file):
    """ Appends to csv
    """
    with open(file, 'a', newline='') as csvfile:
        fieldnames = ['cluster_rank', 'cluster', 'description', 'cohesion', 'novelty']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if(cluster_index == 1):
            writer.writeheader()
        writer.writerow({'cluster_rank': cluster_index - 1,
                         'cluster': data[cluster_index - 1]['cluster'],
                         'description': get['desc'],
                         'cohesion': get['cohesion'],
                         'novelty': get['novelty']})

