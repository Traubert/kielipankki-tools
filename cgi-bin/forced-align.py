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

from kpdemosdev import *

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
    columns = ["start", "end", "word"]
    if ("audio-file" in form and "transcript-file" in form):
        # audio_filename = form["audio-file"].filename
        # transcript_filename = form["transcript-file"].filename
        # audio_fobj = form["audio-file"].filename
        # transcript_filename = form["transcript-file"].filename

        files = {'audio': (form["audio-file"].filename, form["audio-file"].file),
                 'transcript': (form["transcript-file"].filename, form["transcript-file"].file),
                 'Content-Disposition': 'form-data',
                 'Content-Type': 'multipart/form-data'}
        # files = {'file': (filename, inputfile.file, 'application/octet-stream'),
        #          'Content-Disposition': 'form-data; name="file"; filename="' + filename + '"',
        #          'Content-Type': 'application/octet-stream'}

        # audiobytes = inputfile.file.read()
        submit_url = 'http://kielipankki.rahtiapp.fi/audio/align/fi/submit_file'
        try:
            response = requests.post(submit_url, files = files)
        except exception as e:
            result += str(e)
        log(response.text)
        job = json.loads(response.text)
        jobid = job["jobid"]
        target_params = '?job=' + jobid
        time.sleep(1)
    if jobid is not None:
        query_url = 'http://kielipankki.rahtiapp.fi/audio/align/fi/query_job'
        response = requests.post(query_url, data = jobid)
        log(response.text)
        j = json.loads(response.text)
        if "status" in j and j["status"] == "pending":
            result = '<p>In queue..</p>'
        elif "status" in j and j["status"] == "done":
            done = True
            session_key = jobid
            ctm = j["results"]["ctm"]
            eaf = j["results"]["eaf"]
            TextGrid = j["results"]["TextGrid"]
            ctm_words = ctm.split('\n')
            n_words = len(ctm_words)
            preview_words = [word.split() for word in ctm_words[:10]]
            if n_words > 10:
                preview_words.append(['...','...','...'])
            #time_spent = j["processing_finished"] - j["processing_started"]
            #result = '<h3>Full transcript:</h3><div class="text-wrap" style="width: 80rem;"><hr><p class="text">{FULL_TRANSCRIPT}</p><hr></div><p><small>Decoding completed in {TIME_SPENT:.2f} seconds</small></p>'

            result += '''
            <div class="row">
            <div class="col-md-auto py-4">
            <a class="btn btn-info" href="https://kielipankki.fi/tools/demo/kielipankki-tools/tmp/{filename}.ctm" download="align_result.ctm" role="button">Download ctm</a></div>
            <div class="col-md-auto py-4">
            <a class="btn btn-info" href="https://kielipankki.fi/tools/demo/kielipankki-tools/tmp/{filename}.eaf" download="align_result.eaf" role="button">Download eaf</a></div>
            <div class="col-md-auto py-4">
            <a class="btn btn-info" href="https://kielipankki.fi/tools/demo/kielipankki-tools/tmp/{filename}.TextGrid" download="align_result.TextGrid" role="button">Download TextGrid</a></div></div>
            '''.format(html_root = hostname, filename = session_key)


            result += wrap_in_tags('Preview', 'h3') + make_table(preview_words, header = columns) + "\n"
            # for section in json_result["sections"]:
            #     start_time = float(section["start"])
            #     result += wrap_in_tags(wrap_in_tags(section["transcript"], 'p'), 'b')
            #     words = []
            #     for word in section["words"]:
            #         words.append([word["word"], '{:.2f}'.format(start_time + float(word["start"])), '{:.2f}'.format(start_time + float(word["end"]))])
            #         if done:
            #             all_words.append(words[-1])
            #     result += make_table(words, header = columns) + "\n"
            #     transcripts.append(section["transcript"])
            # if done:
            #     result = result.format(FULL_TRANSCRIPT = '\n<br>'.join(transcripts), TIME_SPENT=time_spent)
                
            # if ("status" in j and j["status"] == "pending") or ("done" in j and j["done"] == False):
            #     continue
            # else:
            #     result = json.dumps(j, indent=2)
            #     break
            result_rows = []
        else:
            result = '<p>Got malformed response from server:</p>\n' + wrap_in_tags(json.dumps(j, indent=2), 'p')

        if done:
            write_file(ctm, "ctm", session_key)
            write_file(eaf, "eaf", session_key)
            write_file(TextGrid, "TextGrid", session_key)

        # write_excel([column_names] + out_rows, session_key, "Output from fintag")
            
        # result += "<p>Result:</p>\n"
        # <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.xlsx" download="ner_tagged.xlsx" role="button">Download Excel spreadsheet</a>
        # </div>
        # </div>
        # '''.format(html_root = hostname, filename = session_key)
        # result += make_table(out_rows, header = column_names) + "\n"

    body = wrap_in_tags("forced-align demo", "h2")
    body += '''
<h6>Align Finnish audio with transcript</h6>
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle lead">Entering input</h6>
      <p class="card-text">Upload one audio file and one plain text transcript file. Any format known to ffmpeg may work, but wav and mp3 have been tested.
      </p>
      <h6 class="card-subtitle lead">Understanding output</h6>
      <p class="card-text">
      </p>
    </div>
  </div>
</div>
<form method="post" action="/cgi-bin/{scriptname}" enctype="multipart/form-data">
  <div class="form-group">
    <div class="row">
      <div class="col-md-auto pl-0 border">
        <label for="submit_audio_file">Upload an audio file (wav, mp3, ...)</label>
        <input type="file" class="form-control-file" name="audio-file" id="submit_audio_file">
      </div>
    </div>
    <div class="row">
      <div class="col-md-auto pl-0 border">
        <label for="submit_transcript_file">Upload a plain text transcript (txt)</label>
        <input type="file" class="form-control-file" name="transcript-file" id="submit_transcript_file">
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
        head = make_head(title = 'align demo')
    else:
        head = make_head(title = 'align demo', extra_meta = '<meta http-equiv="refresh" content="2; url=forced-align.py{refresh_target}">'.format(refresh_target = target_params))
    sys.stdout.buffer.write(wrap_html(head, wrap_in_tags(body, 'div', attribs='class="container pt-1"', oneline = False)).encode("utf-8"))

if __name__ == '__main__':
    try:
        print_content()
    except Exception as ex:
        log(str(ex))
