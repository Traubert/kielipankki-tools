#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import cgi
import time
import hashlib
import os
import cgitb
cgitb.enable()

from kpdemos import *

def print_content():
    time_start = time.time()
    clean_tempfiles()
    form = cgi.FieldStorage()
    result = ""
    if "file" in form and form["file"].filename != "":
        try:
            output, session_key, filename = ocr_from_file(form["file"], form["lang"].value)
        except Exception as ex:
            log(str(ex))
        write_docx(output.decode("utf-8"), session_key, "OCR result")
        result += "<p>Result:</p>\n"
        result += '''
        <div class="row">
        <div class="col-md-auto">
        <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.docx" download="ocr_result.docx" role="button">Download .docx</a>
        </div>
        </div>
        '''.format(html_root = hostname, filename = session_key)
        if not filename.endswith("pdf"):
            result += '<img src=' + hostname + '/kielipankki-tools/tmp/' + filename + '/>\n'
        result += text_to_html(output.decode("utf-8"))
    body = wrap_in_tags("OCR with tesseract demo", "h2")
    body += '<h6>Recognize text from images in multiple languages.</h6>'
    body += '''
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <p class="card-text">
        Select an image (gif, jpg, png or tiff) or PDF containing images on your computer to upload, and text in it will be recognized using <a href="https://github.com/tesseract-ocr">tesseract</a> with language settings from the dropdown box.
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
<select class="bootstrap-select" name="lang">
{select_langs}
</select>
      </div>
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
'''.format(select_langs = open('tesseract_langs_select.txt', encoding="utf-8").read() , scriptname = os.path.basename(sys.argv[0]), content = result, TIME_SPENT = time.time() - time_start)

    sys.stdout.buffer.write(wrap_html(make_head(title = 'OCR with tesseract demo'), wrap_in_tags(body, 'div', attribs='class="container pt-1"', oneline = False)).encode("utf-8"))

if __name__ == '__main__':
    print_content()
