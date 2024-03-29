import os
import csv
import json
from datetime import datetime

from home.utils import half_norm_bins, sample_bins, add_playable_links, add_playable_links, append_to_csv

from django.shortcuts import render
from random import shuffle
from django.urls import path
from django.views.generic import TemplateView, RedirectView
from scipy.stats import halfnorm

from google.cloud import storage
from django.conf import settings

class LoginView(TemplateView):
    template_name = "login.html"
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_%H:%M")
        context['date_time'] = date_time
        return context



class HomeView(TemplateView):
    template_name = "home.html"
    sample_size = 10
    no_clusters = 15
    test_no = 1

    def get_context_data(self, **kwargs):
        # context = super().get_context_data(**kwargs)
        # with open(os.path.join(settings.BASE_DIR, 'home/json/summary.json'), "r") as read_file:
        #     data = json.loads(read_file)
        print(self.request.GET)

        context = super().get_context_data(**kwargs)
        json_path = os.path.join(settings.BASE_DIR, 'home/json/')
        # with open("{0}{1}".format(json_path, "20.json"), "r") as read_file:
        with open("{0}{1}".format(json_path, "combination_three.json"), "r") as read_file:
            data = json.load(read_file)

        # get the best cluster first if we enter the page without a cluster_index
        # and set the ssion
        cluster_index = 0
        if 'cluster_index' in self.request.GET:
            cluster_index = int(self.request.GET['cluster_index'])
        else:
            write_file = open("{0}{1}".format(json_path, "combination_three.json"), "w+")
            # shuffle(data)
            write_file.write(json.dumps(data))
            # flush the session and set it


            # make bucket folder based on date-session

        # if you are not the first index append the previous result to a csv
        # appends locally
        # upload the updated file

        if(cluster_index is not 0 and 'is_button' not in self.request.GET):
            file = "{0}result/{1}-{2}".format(json_path,
                                              self.request.GET['name'],
                                              self.request.GET['date_time'])

            append_to_csv(get=self.request.GET, cluster_index=cluster_index, data=data, file=file)

            # TODO: better way to store the key
            client = storage.Client.from_service_account_json('syb-master-thesis-content-a0efc8ae60c6.json')
            bucket = client.get_bucket('syb-cluster-evaluation')
            blob = bucket.blob("Results/{0}/{1}.csv".format(self.request.GET['name'], self.request.GET['date_time']))
            blob.upload_from_filename(file)

        # if cluster_index is larger we are done with the evaluation
        if(cluster_index < self.no_clusters):
            # get our relevant cluster, the ith smallest
            data = data[cluster_index]
            if(data['type'] != 'playlist'):
                # if we have many songs draw a sample * 3 from a half norm distribution
                # the scalar is supposed to allow the sample size when we discard some songs later
                if(len(data['songs']) > self.sample_size * 3):
                    bins = half_norm_bins(data['songs'])
                    sample = sample_bins(bins, sample_size=self.sample_size * 3)
                else:
                    sample = data['songs']
            else:
                sample = data['songs'][:self.sample_size*3]

            sample = add_playable_links(sample)

            if(data['type'] != 'playlist'):
                # now we can actually take the sample size and be almost sure we get the amount of songs we want
                # do it again, to force the ditribution and not slice
                bins = half_norm_bins(sample)
                sample = sample_bins(bins, sample_size=self.sample_size)

            else:
                # for other we don't care if we slice or not
                sample = sample[:self.sample_size]

            context['cluster_index'] = cluster_index
            context['sample'] = sample
            context['no_clusters'] = self.no_clusters
            context['date_time'] = self.request.GET['date_time']
            context['name'] = self.request.GET['name']
            context = {**context, **data}
        return context
