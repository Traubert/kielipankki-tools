#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import cgi
import time
import hashlib
import os
import cgitb
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

column_names = ["Surface form", "Lemma", "Named entity tag", "Part of Speech"]

def print_content():
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
        process = Popen([wrkdir + "/run-nertag"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = process.communicate(input=inputval)
        session_key = hashlib.md5(out).hexdigest()
        out_rows = permute_rows(tsv2rows(out.decode("utf-8")), (0, 1, 3, 2))
#        orig_rows = tsv2rows(out.decode("utf-8"))
#        out_rows = orig_rows
        write_excel([column_names] + out_rows, session_key, "Output from fintag")
        write_tsv(make_tsv([column_names] + out_rows), session_key)
        result += "<p>Result:</p>\n"
        result += '''
        <div class="row">
        <div class="col-md-auto">
        <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.tsv" download="ner_tagged.tsv" role="button">Download TSV</a>
        <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.xlsx" download="ner_tagged.xlsx" role="button">Download Excel spreadsheet</a>
        </div>
        </div>
        '''.format(html_root = hostname, filename = session_key)
        result += make_table(out_rows, header = column_names) + "\n"

    body = wrap_in_tags("fintag demo", "h2")
    body += '''
<h6>Annotate running text with FinnPos and FiNER.</h6>
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle mb-2">Entering input</h6>
      <p class="card-text">
        You have a choice between three options: enter text in the text box, choose a demo text, or upload a file. A variety of file formats are supported: plain utf-8 text (.txt), and unless the formatting is especially convoluted, .pdf, .doc, .docx, .csv, .epub, .html, .odt, .rtf and .xls files.
      </p>
      <h6 class="card-subtitle mb-2">Understanding output</h6>
      <p class="card-text">
        The output is presented as a table, which is also available for download as a spreadsheet or TSV (tab separated values) file.
      </p>
      <p class="card-text">
        The table has five columns. The first shows the token (word, punctuation unit, url... whatever the tokenizer consideres to be one token) as it appeared in the original text. The second column shows the lemma, or base form, of the token. The next column shows the most likely morphological tags for the token. The final two columns represent named entities; the first one contains tags for persons, organisations and locations, the second contains everything else (mostly time expressions and events).
      </p>
    </div>
  </div>
</div>
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
<p><small>Page generated in {TIME_SPENT:.2f} seconds</small></p>
'''.format(scriptname = os.path.basename(sys.argv[0]), content = result, TIME_SPENT = time.time() - time_start)

    sys.stdout.buffer.write(wrap_html(make_head(title = 'fintag demo', scripts = (populate_js,)), wrap_in_tags(body, 'div', attribs='class="container-fluid"', oneline = False)).encode("utf-8"))

if __name__ == '__main__':
    print_content()
