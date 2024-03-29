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

    On the second call we make sure we have the sample size of songs
    """
    # number of samples we want from each bin
    bin_counts = [round(0.6827 * sample_size), round((0.9545 - 0.6827) * sample_size), round((1 - 0.9545) * sample_size)]
    prev_k = 0
    sample = []

    for i in reversed(range(0,len(bins))):
        if(len(bins[i]) >= bin_counts[i] + prev_k):
            k = bin_counts[i] + prev_k
            prev_k = 0
        elif(len(bins[i]) >= bin_counts[i]):
            k = bin_counts[i] + prev_k
        else:
            k = len(bins[i])
            prev_k = (bin_counts[i] - len(bins[i])) + prev_k

        sample_from_bin = random.sample(bins[i], k=k)
        ## could be better
        sample = sort_sample(sample_from_bin) + sample

    return sample


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
               'Authorization': 'Bearer eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE1NTg5NDk1MTIsImlhdCI6MTU1NjU0ODgyMywic3ViIjoiVlhObGNpd3NNV2syTkRBemNYWjRPR2N2IiwidHlwIjoic3VwZXJ1c2VyIn0.JOYGcpQ6fZfZp-rGdfJ62Os2zA9CM4bjHr3wUit3BHzC2GdgJJbwbmHmrLsWZSWKcuPQmcWX0ZqWr6R-zbng-w'
               }

    get_request = requests.post(
        url=ext_url, data=json.dumps(tracks), headers=headers)
    # print(get_request)
    get_json = get_request.json()

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
        fieldnames = ['cluster_rank', 'cluster', 'type', 'description', 'quality', 'cohesion', 'culture', 'novelty']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if(cluster_index == 1):
            writer.writeheader()
        writer.writerow({'cluster_rank': cluster_index - 1,
                         'cluster': data[cluster_index - 1]['cluster'],
                         'type': data[cluster_index - 1]['type'],
                         'description': get['desc'],
                         'quality': get['quality'],
                         'cohesion': get['cohesion'],
                         'culture': get['culture'],
                         'novelty': get['novelty']})

