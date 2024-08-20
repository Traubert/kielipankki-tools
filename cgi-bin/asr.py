#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import cgi
import time
import os
import requests
import json
import cgitb
cgitb.enable()
#from tempfile import TemporaryFile
#import urllib
import hashlib

from kpdemos import *

def print_content():
    jobid = None
    done = False
    query = os.environ.get("QUERY_STRING")
    target_params = ''
    if query is not None:
        target_params = '?' + query
        query_dict = {}
        if '=' in query:
            query_dict = dict(map(lambda x: x.split('='), query.split('&')))
        if 'job' in query_dict:
            jobid = query_dict["job"]
    # # urllib.parse(url).query

    time_start = time.time()
    clean_tempfiles()
    form = cgi.FieldStorage()
    result = ""
    inputval = ""
    columns = ["word", "start", "end"]
    if ("file" in form and form["file"].filename != ""):
        filename = form["file"].filename
        inputfile = form["file"]
        files = {'file': (filename, inputfile.file),
                          'Content-Disposition': 'form-data; name="file"; filename="' + filename + '"',
                          'Content-Type': 'multipart/form-data'}
        # files = {'file': (filename, inputfile.file, 'application/octet-stream'),
        #          'Content-Disposition': 'form-data; name="file"; filename="' + filename + '"',
        #          'Content-Type': 'application/octet-stream'}

        # audiobytes = inputfile.file.read()
        submit_url = 'http://kielipankki.rahtiapp.fi/audio/asr/fi/submit_file'
        # submit_url = 'http://kielipankki.rahtiapp.fi/audio/asr/fi/segmented'
        try:
            response = requests.post(submit_url, files = files)
        except exception as e:
            result += str(e)
        job = json.loads(response.text)
        jobid = job["jobid"]
        target_params = '?job=' + jobid
        time.sleep(1)
    if jobid is not None:
        query_url = 'http://kielipankki.rahtiapp.fi/audio/asr/fi/query_job/tekstiks'
        response = requests.post(query_url, data = jobid)
        j = json.loads(response.text)
        if "done" not in j:
            if "status" in j and j["status"] == "pending":
                result = '<p>In queue..</p>'
            else:
                result = '<p>Got malformed response from server:</p>\n' + wrap_in_tags(json.dumps(j, indent=2), 'p')
        else:
            if j["done"] == False:
                result = '<p>Partial result, wait for more..</p>\n'
            else:
                done = True
                all_words = []
                session_key = jobid
                time_spent = j["processing_finished"] - j["processing_started"]
                result = '<h3>Full transcript:</h3><div class="text-wrap" style="width: 80rem;"><hr><p class="text">{FULL_TRANSCRIPT}</p><hr></div><p><small>Decoding completed in {TIME_SPENT:.2f} seconds</small></p>'
                result += '''
                <div class="row">
                <div class="col-md-auto py-4">
                <a class="btn btn-info" href="https://kielipankki.fi/tools/demo/kielipankki-tools/tmp/{filename}.tsv" download="asr_result.tsv" role="button">Download TSV</a></div></div>
            '''.format(html_root = hostname, filename = session_key)

            json_result = j["result"]
            transcripts = []
            for section in json_result["sections"]:
                start_time = float(section["start"])
                result += wrap_in_tags(wrap_in_tags(section["transcript"], 'p'), 'b')
                words = []
                for word in section["words"]:
                    words.append([word["word"], '{:.2f}'.format(start_time + float(word["start"])), '{:.2f}'.format(start_time + float(word["end"]))])
                    if done:
                        all_words.append(words[-1])
                result += make_table(words, header = columns) + "\n"
                transcripts.append(section["transcript"])
            if done:
                result = result.format(FULL_TRANSCRIPT = '\n<br>'.join(transcripts), TIME_SPENT=time_spent)
                
            # if ("status" in j and j["status"] == "pending") or ("done" in j and j["done"] == False):
            #     continue
            # else:
            #     result = json.dumps(j, indent=2)
            #     break
        result_rows = []

        if done:
            write_tsv(make_tsv([columns] + all_words), session_key)

        # write_excel([column_names] + out_rows, session_key, "Output from fintag")
            
        # result += "<p>Result:</p>\n"
        # <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.xlsx" download="ner_tagged.xlsx" role="button">Download Excel spreadsheet</a>
        # </div>
        # </div>
        # '''.format(html_root = hostname, filename = session_key)
        # result += make_table(out_rows, header = column_names) + "\n"

    body = wrap_in_tags("asr demo", "h2")
    body += '''
<h6>Recognise Finnish speech with Kaldi and Aalto-asr</h6>
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle lead">Entering input</h6>
      <p class="card-text">Currently, only file uploads are supported. Any format known to ffmpeg may work, but wav and mp3 have been tested.
      </p>
      <h6 class="card-subtitle lead">Understanding output</h6>
      <p class="card-text">The audio is split into chunks separated by silence. These chunks are processed separately, in parallel. The output shows them in the correct order. Tabular output shows <ol><li>The full recognized text, once it is ready</li><li>The recognized chunks, as they are completed</li><li>A table with each word in the chunk, with time information</li></ol>
      </p>
      <p class="card-text">When results are complete, a tsv file with all the timing information is generated for downloading.
      </p>
    </div>
  </div>
</div>
<form method="post" action="/cgi-bin/{scriptname}" enctype="multipart/form-data">
  <div class="form-group">
    <div class="row">
      <div class="col-md-auto pl-0">
        <input type="file" class="form-control-file" name="file" id="submit_file">
      </div>
    </div>

    <div class="row">
      <div class="col-md-auto">
        <button type="submit" class="btn btn-primary" id="submit_button">Submit</button>
      </div>
    </div>

  </div>
</form>
<div class="row">
  <div class="col-md-auto">
    {content}
  </div>
</div>
<p><small>Page generated in {TIME_SPENT:.2f} seconds</small></p>
'''.format(scriptname = os.path.basename(sys.argv[0]), content = result, TIME_SPENT = time.time() - time_start)
    if jobid is None or done == True:
        head = make_head(title = 'asr demo')
    else:
        head = make_head(title = 'asr demo', extra_meta = '<meta http-equiv="refresh" content="2; url=asr.py{refresh_target}">'.format(refresh_target = target_params))
    sys.stdout.buffer.write(wrap_html(head, wrap_in_tags(body, 'div', attribs='class="container pt-1"', oneline = False)).encode("utf-8"))

if __name__ == '__main__':
    print_content()
