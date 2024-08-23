#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import cgi
import time
import hashlib
import os
import cgitb
import hfst
cgitb.enable()

from kpdemosdev import *

populate_js = '''
function populateTextField() {
document.getElementById("inputted_text").value = "Urho Kaleva Kekkonen (3. syyskuuta 1900 Pielavesi – 31. elokuuta 1986 Helsinki) oli suomalainen poliitikko ja kahdeksas Suomen tasavallan presidentti. Hän oli tasavallan istuva presidentti yhtäjaksoisesti vuodesta 1956 alkuvuoteen 1982, yli 25 vuoden ajan. Viimeinen presidenttikausi jäi kesken sairauden takia. Kekkonen on Suomen historian pitkäaikaisin presidentti. Hän on ensimmäinen henkilö, joka toimi tasavallan presidenttinä kaksi kokonaista kautta ja ainoa, joka on valittu toimeensa useammaksi kuin kahdeksi kaudeksi, mikä ei perustuslakiin eli entiseen Suomen hallitusmuotoon myöhemmin tehdyn muutoksen ja nykyisen Suomen perustuslain mukaan olisi enää mahdollistakaan. Ennen presidenttiyttään Kekkonen toimi muun muassa juristina, yleisurheilijana, oikeusministerinä, eduskunnan puhemiehenä sekä viiden hallituksen pääministerinä. Presidentin valitsijamiehenä, kansanedustajana ja ministerinä Kekkonen oli valitsemassa Kyösti Kalliota vuonna 1937, Risto Rytiä vuosina 1940 ja 1943, Gustaf Mannerheimia vuonna 1944 sekä J. K. Paasikiveä vuonna 1946.";
}
'''

# download_js = '''
# function download(filename, text) {
#   var element = document.createElement('a');
#   element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
#   element.setAttribute('download', filename);

#   element.style.display = 'none';
#   document.body.appendChild(element);

#   element.click();

#   document.body.removeChild(element);
# }
# '''

column_names = ["Token", "Analysis"]

def print_content():
    log("starting")
    time_start = time.time()
    clean_tempfiles()
    form = cgi.FieldStorage()
    result = ""
    inputval = ""
    if "input" in form:
        inputval = form["input"].value.encode("utf-8")
    if "file" in form and form["file"].filename != "":
        inputval = text_from_file(form["file"])
    if inputval != "":
        inputval = inputval.decode("utf-8").lower()
        log("got input " + inputval)
        pmatcher = hfst.PmatchContainer("ofitwol.hfst")
        log("loaded pmatch")
        session_key = hashlib.md5(("ofitwol" + inputval).encode("utf-8")).hexdigest()
        locationvectorvector = pmatcher.locate(inputval)
        log("ran pmatch")
        out_rows = []
        out_utf8 = ""
        for locationvector in locationvectorvector:
            if locationvector[0].output == '@_NONMATCHING_@':
                continue
            out_rows.append([locationvector[0].input, locationvector[0].output])
            out_utf8 += locationvector[0].input + '\t' + locationvector[0].output + '\n'
        excel_download_button_string = '<a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.xlsx" download="ofitwol.xlsx" role="button">Download Excel spreadsheet</a>'.format(html_root = hostname, filename = session_key)
        try:
            write_excel(out_rows, session_key, "Output from fintok")
        except Exception as ex:
            excel_download_button_string = '<a class="btn btn-info" role="button">Writing Excel spreadsheet failed!</a>'
        write_txt(out_utf8, session_key)
        # result += "<p>Result: {} tokens, {} sentences</p>\n".format(len(out_rows), len(list(filter(lambda x: x[0] == "", out_rows))))
        result += '''
        <div class="row">
        <div class="col-md-auto">
        <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.txt" download="tokenized.txt" role="button">Download text</a>
        {excel_button_text}
        </div>
        </div>
        '''.format(html_root = hostname, filename = session_key, excel_button_text = excel_download_button_string)
        result += make_table(out_rows, header = column_names) + "\n"
    body = wrap_in_tags("ofitwol demo", "h2")
    body += '<h6>Analyze with <a href="https://pytwolc.readthedocs.io/en/latest/ofitwol.html">OFITWOL</a></h6>'
    body += '''
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle lead">Entering input</h6>
      <p class="card-text">
        You have a choice between three options: enter text in the text box, choose a demo text, or upload a file. A variety of file formats are supported: plain utf-8 text (.txt), and unless the formatting is especially convoluted, .pdf, .doc, .docx, .csv, .epub, .html, .odt, .rtf and .xls files.
      </p>
      <p class="card-text">Because OFITWOL is limited to lower-case letters, the input is converted to lowercase before analysis.</p>
      <h6 class="card-subtitle lead">Understanding output</h6>
      <p class="card-text">
        The output is presented as one line per token, with content (eg. punctuation) not recognised by OFITWOL suppressed.</p>
      <p class="card-text">OFITWOL implements a morphophonemic analyser (ofimphon) and a guesser (ofiguess). If ofimphon knows a word, its output (the one with lowest weight, if multiple) is reported, otherwise the output of ofiguess (again, with lowest weight) is reported.</p>
      <p class="card-text">Output may also be downloaded as text or a spreadsheet.</p>
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

    sys.stdout.buffer.write(wrap_html(make_head(title = 'fintok demo', scripts = (populate_js,)), wrap_in_tags(body, 'div', attribs='class="container pt-1"', oneline = False)).encode("utf-8"))

if __name__ == '__main__':
    print_content()
