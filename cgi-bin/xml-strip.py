#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import cgi
import time
import hashlib
import os
import cgitb
cgitb.enable()
from lxml import etree

from kpdemosdev import *

def print_content():
    time_start = time.time()
    clean_tempfiles()
    form = cgi.FieldStorage()
    result = ""
    inputval = ""
    fake_toplevel_tags = True
    if "input" in form:
        inputval = form["input"].value.strip()
    if ("file" in form and form["file"].filename != "") or inputval != "":
        out = ""
        exception_string = ""
        if "file" in form and form["file"].filename != "":
            inputval = str(form["file"].file.read(), encoding = 'utf-8').strip()
        if (not inputval.startswith('<')) and fake_toplevel_tags:
            inputval = '<?xml version="1.0" encoding="UTF-8"?>\n<kielipankki>\n' + inputval + '\n</kielipankki>'
        try:
            tree = etree.fromstring(inputval.encode('utf-8'))
            out = etree.tostring(tree, encoding = 'unicode', method = 'text').strip()
            # if "file" in form and form["file"].filename != "":
            #     tree = etree.parse(form["file"].file)
            # else:
            #     tree = etree.fromstring(inputval)
            # out = etree.tostring(tree, encoding = 'unicode', method = 'text')
        except Exception as ex:
            exception_string = str(ex)
        if out == "":
            if exception_string != "":
                result += "<p>XML parsing failed with the following exception: {}</p>\n".format(exception_string)
            else:
                result += "<p>Got empty result, no errors</p>\nInput was\n{}".format(inputval)
        else:
            session_key = hashlib.md5(out.encode('utf-8')).hexdigest()
#            out_html = make_table([text_to_html(inputval), text_to_html(out)])
            out_html = text_to_html(out)
            write_txt(out, session_key)
            result += out_html
            result += '''
            <div class="row">
            <div class="col-md-auto">
            <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.txt" download="xml_stripped.txt" role="button">Download text</a>
            </div>
            </div>
            '''.format(html_root = hostname, filename = session_key)
    body = wrap_in_tags("xml validation and stripping", "h2")
    body += '<h6>Validate XML and strip tags from it.</h6>'
    body += '''
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle lead">Entering input</h6>
      <p class="card-text">
        Enter XML text in the text box, or upload a file.
      </p>
      <h6 class="card-subtitle lead">Output</h6>
      <p class="card-text">
        If the XML is valid, the text content is written as formatted HTML, and saved to a .txt file. Otherwise, the first error encountered is reported.
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
        <textarea name="input" type="text" rows="10" cols="60" class="form-control" id="inputted_text" placeholder="XML goes here."></textarea>
      </div>
    </div>
    <div class="row">
      <div class="col-md-auto">
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

    sys.stdout.buffer.write(wrap_html(make_head(title = 'xml-strip demo', scripts = ()), wrap_in_tags(body, 'div', attribs='class="container pt-1"', oneline = False)).encode("utf-8"))

if __name__ == '__main__':
    print_content()
