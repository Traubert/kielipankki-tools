#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import subprocess
import cgi
import time
import os
import cgitb
cgitb.enable()
import string
import hashlib

wrkdir = "/var/www/kielipankki-tools"

sys.path.append('/var/www/cgi-bin')
from kpdemosdev import *


def make_doctype():
    return "Content-type: text/html\n\n<!doctype html>\n"

def wrap_in_tags(content, tag, oneline = True, attribs = None):
    if attribs == None:
        attribs = ""
    else:
        attribs = " " + attribs
    if oneline:
        return "<" + tag + attribs + ">" + content + "</" + tag + ">\n"
    else:
        return "<" + tag + attribs + ">\n" + content + "\n</" + tag + ">\n"


demo_js = '''
function populateTextField() {
document.getElementById("inputted_text").value = "Tosi hyvä juttu! Tosi huono juttu! Elokuva oli miellyttävä. Elokuva oli puuduttava. Elokuva oli ihan ok.";
}
'''
def print_content():
    remember_input_js = ''
    time_start = time.time()
    form = cgi.FieldStorage()
    result = ""
    inputval = ""
    model = "verkkokauppa"
    if "input" in form:
        inputval = form["input"].value
    if "file" in form and form["file"].filename != "":
        inputval = text_from_file(form["file"]).decode("utf-8")
    if inputval != '':
        model = form["model_selection"].value
        if model == "no_selection":
            model = "Social media manually annotated model"
        if model == 'Social media smiley model':
            import smiley_predict
            predict = smiley_predict.predict
        elif model == 'Social media manually annotated model':
            import s24_majority_vote
            predict = s24_majority_vote.predict
        else:
            import sentiment
            predict = sentiment.predict
            
        remember_input_js = '''
        window.onload = function() {{
        document.getElementById('inputted_text').value = {};
        }};
        '''.format(json.dumps(form["input"].value))
        try:
            sentences = tokenize(inputval)
            if len(sentences) > 1:
                sentences.append(sum(sentences, []))
            session_key = hashlib.md5(inputval.encode("utf-8")).hexdigest()
#            full_text = ' '.join(map(lambda x: ' '.join(x), sentences))
            results = predict(sentences)
#            sys.stdout.buffer.write(results.__str__().encode("utf-8"))
#            sys.stdout.buffer.write(wrap_html(make_head("finsentiment demo"), wrap_in_tags(wrap_in_tags(str(sentences) + str(results), "body", oneline = False), 'div', attribs='class="container-fluid"')).encode("utf-8"))
#            return
            texts_and_results = list(zip(results, map(lambda x: ' '.join(x), sentences)))
            result += "<p>Result:</p>\n"
            result += '''
<div class="row">
  <div class="col-md-auto">
    <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.tsv" download="finnsentiment_result.tsv" role="button">Download TSV</a>
    <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.xlsx" download="finnsentiment_result.xlsx" role="button">Download Excel spreadsheet</a>
  </div>
</div>
            '''.format(html_root = hostname, filename = session_key)
            column_names = ("Sentiment", "Text")
            result += make_table(texts_and_results, header = column_names) + "\n"
            write_tsv(make_tsv(texts_and_results), session_key)
            write_excel(texts_and_results, session_key, "Output from FinnSentiment")
        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            errinfo = "<p>" + str(sentences) + "</p>"
            sys.stdout.buffer.write(wrap_html(make_head("FinnSentiment demo error"), wrap_in_tags(errinfo + "<p>Got exception " + str(ex) + "</p>\n", "body", oneline = False)).encode("utf-8"))
            return
        

    body = ""
    body += wrap_in_tags("FinnSentiment demo", "h2")
    body += '<h6>Estimate the sentiment of texts and sentences.</h6>'
    body += '''
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle mb-2"></h6>
      <p class="card-text">
        This demo estimates a sentiment (positive, negative, or neutral) for each sentence in the input text, and also for the input text as a whole.
      </p>
      <p class="card-text">
        The sentiment analysis relies on three resources:
        <ol>
          <li>
            Word embeddings calculated from a corpus of Finnish text.
          </li>
          <li>
            Product reviews harvested from the Internet.
          </li>
          <li>
            A word-based convolutional neural network with 100 kernels each of sizes 2, 3, 4 and 5 words. The neural network is trained to predict the rating associated with product reviews, and the prediction it gives to the input text is converted to a sentiment.
          </li>
        </ol>
      </p>
    </div>
  </div>
</div>
<form method="post" action="/cgi-bin/predict/{scriptname}" enctype="multipart/form-data">
  <div class="form-group">
    <div class="row">
      <div class="col-md-auto">
        <select class="custom-select" name="model_selection">
          <option name="no_selection">Select model</option>
          <option name="s24_majority_vote">Social media manually annotated model</option>
          <option name="verkkokauppa">Product review model</option>
          <option name="s24_smiley">Social media smiley model</option>
        </select>
      </div>
    </div>
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
  <div class="col-md-8">
    {content}
  </div>
</div>
<div class="row"><hr /></div>
<div class="row">
  <div class="col-6">
    <div class="card">
      <div class="card-body">
        <h5 class="card-title">
          Citing this service
        </h5>
	<div class="card-text">
          <p>The model titled "Social media manually annotated model" has been described in a peer-reviewed article. If you use it, you may cite it as:</p>
          <small><p>Lindén, K., Jauhiainen, T. & Hardwick, S. FinnSentiment: a Finnish social media corpus for sentiment polarity annotation. <i>Lang Resources & Evaluation</i> <b>57</b>, 581–609 (2023). <a href="https://doi.org/10.1007/s10579-023-09644-5">https://doi.org/10.1007/s10579-023-09644-5</a></p></small>
        </div>
      </div>
    </div>
  </div>
</div>

<p><small>Page generated in {TIME_SPENT:.2f} seconds</small></p>
'''.format(scriptname = os.path.basename(sys.argv[0]), content = result, TIME_SPENT = time.time() - time_start)# + str(form)
    sys.stdout.buffer.write(wrap_html(make_head("FinnSentiment demo", scripts = (remember_input_js, demo_js)), wrap_in_tags(wrap_in_tags(body, "body", oneline = False), 'div', attribs='class="container pt-1"')).encode("utf-8"))

if __name__ == '__main__':
    try:
        print_content()
    except Exception as ex:
        log(str(ex))
