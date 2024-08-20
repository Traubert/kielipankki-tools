#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import cgi
import time
import hashlib
import os
import requests
import json
import cgitb
cgitb.enable()

from kpdemosdev import *

def html_from_conllu(conllu):
    kielipankki_conllu_api_url = 'http://kielipankki.2.rahtiapp.fi/utils/conllu2html'
    if type(conllu) == bytes:
        encoded_text = text
    else:
        encoded_text = conllu.encode('utf-8')
    response = requests.post(kielipankki_conllu_api_url,
                             data = encoded_text,
                             params = {})
    response.raise_for_status()
    return response.text

def conllu_from_text(text):
    kielipankki_parse_api_url = 'http://kielipankki.2.rahtiapp.fi/text/fi/parse'
    if type(text) == bytes:
        encoded_text = text
    else:
        encoded_text = text.encode('utf-8')
    response = requests.post(kielipankki_parse_api_url,
                             data = encoded_text,
                             params = {})
    response.raise_for_status()
    return response.text

populate_js = '''
function populateTextField() {
document.getElementById("inputted_text").value = "Se, joka oli löytänyt miehen, jota ilman olisimme eksyneet, löytyi itse.";
}
'''

remember_js = ''

def print_content():
    time_start = time.time()
    clean_tempfiles()
    form = cgi.FieldStorage()
    result = ""
    inputval = ""
    if "input" in form:
        inputval = form["input"].value.encode("utf-8")
        # try:
        #     remember_js = 'function rememberInput() { document.getElementById("inputted_text").value = "{_input}"; }'.format(_input = form["input"].value)
        # except Exception as ex:
        #     log(str(ex))

    if "file" in form and form["file"].filename != "":
        inputval = text_from_file(form["file"])
    if inputval != "":
        conllu = conllu_from_text(inputval)
        html = html_from_conllu(conllu)
        session_key = hashlib.md5(conllu.encode("utf-8")).hexdigest()
        write_file(html, 'html', session_key)
        result += '''
        <iframe id="parseResult" title="Parse result" width="1200px" height="720px" scrolling="auto" src="{parsed_html_file}">
        </iframe>
        '''.format(parsed_html_file = 'https://kielipankki.fi/tools/demo/kielipankki-tools/tmp/' + session_key + '.html')
    body = wrap_in_tags("finnish-parse demo", "h2")
    body += '<div class="row"><div class="col-6"><h6>Parse running Finnish text using <a href="http://turkunlp.org/Turku-neural-parser-pipeline/">TurkuNLP\'s TNPP</a>, and visualise with <a href="https://github.com/rug-compling/conllu-viewer">CoNLL-U viewer</a> from The University of Groningen</a></h6></div></div>'
    body += '''
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle lead">Entering input</h6>
      <p class="card-text">
        Enter running text in the text box below. You can use multiple paragraphs, and the output will preserve information about paragraph breaks.
      </p>
      <h6 class="card-subtitle lead">Understanding output</h6>
      <p class="card-text">
      The text is first parsed into a dependency parse tree in CoNLL-U format, and then visualised with dependency arrows that connect words in a sentence with each other. The result area has its own scrollbars you can use to see extra-long sentences.
      </p>
      <p>
      Hovering on arrows highlights the source and target tokens for the arrow, and hovering on tokens brings up a tooltip with the token's morphology and other parse data.
      </p>
    </div>
  </div>
</div>
'''
    body += '''
<form method="post" action="/cgi-bin/{scriptname}" enctype="multipart/form-data">
  <div class="form-group">
    <div class="row">
      <div class="col-md-auto">
        <textarea name="input" type="text" rows="10" cols="60" class="form-control" id="inputted_text" placeholder="Running text goes here."></textarea>
      </div>
    </div>
    <div class="row">
      <div class="col-md-auto">
        <button type="button" class="btn btn-default" onclick="populateTextField()"><b>Or</b> populate with demo text</button>
      </div>
      <div class="col-md-auto"><p><b>Or</b></p></div>
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
<div class="row">
  <div class="col-md-auto">
    <p><small>Page generated in {TIME_SPENT:.2f} seconds</small></p>
  </div>
</div>
'''.format(scriptname = os.path.basename(sys.argv[0]), content = result, TIME_SPENT = time.time() - time_start)

    sys.stdout.buffer.write(wrap_html(make_head(title = 'finparse demo', scripts = (populate_js, remember_js)), wrap_in_tags(body, 'div', attribs='class="container pt-1"', oneline = False), onload = "rememberInput()").encode("utf-8"))

if __name__ == '__main__':
    print_content()
