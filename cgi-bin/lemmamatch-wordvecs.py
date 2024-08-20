#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, sys
from subprocess import Popen, PIPE
import cgi
import time
import hashlib

sys.path.append('/var/www/kielipankki-tools/')
from make_pmscript import make_script
import requests
import json

import cgitb
cgitb.enable()

from kpdemos import *

space_eating_punct = '.!?,;'

def wrap_noun(content):
    return wrap_in_tags(content, "span", attribs = 'style="background-color:DodgerBlue;"').strip()
def wrap_verb(content):
    return wrap_in_tags(content, "span", attribs = 'style="background-color:MediumSeaGreen;"').strip()
def wrap_adjective(content):
    return wrap_in_tags(content, "span", attribs = 'style="background-color:Orange;"').strip()
def wrap_pronoun(content):
    return wrap_in_tags(content, "span", attribs = 'style="background-color:Violet;"').strip()

scripts = (
'''
function populateTextField() {
document.getElementById("inputted_text").value = "Moottoritiellä on kaksi keskikaistan tai kaiteen toisistaan erottamaa ajorataa, joilla ei ole avattavia siltoja. Risteävän liikenteen tulee kulkea eri tasossa kuin moottoritieliikenteen. Moottoritiehen liittyvä ja sieltä poistuva liikenne on johdettava moottoritien alkamis- tai päättymiskohtien taikka tiesuunnitelmassa osoitettujen liittymien kautta. Moottoritien ja siihen rajoittuvan kiinteistön välillä ei saa olla muuta kulkuyhteyttä.\\n\\nMillainen kännykkäliittymä kannattaisi ostaa? Entä mikä operaattori olisi paras juuri minulle? Ja pitäisikö nykyinen liittymä vaihtaa uuteen, halvempaan ja kattavampaan?";
document.getElementById("lemmalist_text").value = "LiittymäKännykkä\\n[Like(liittymä, kännykkä)^500]\\n\\nLiittymäEiKännykkä\\n[Unlike(liittymä, kännykkä)^500]";
$('#lemmalist_collapse').show()
}
''',
'''
function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}
''')

def process_form(form):
    debuginfo = ""
    want_verbs = False
    want_nouns = False
    want_adjectives = False
    want_pronouns = False
    vectorfile = "all-vec.bin"
    if "input" not in form:
        return debuginfo
    session_key = hashlib.md5(form["input"].value.encode("utf-8"))
    matchers = []
    if "fi_nouns" in form:
        want_nouns = True
    if "fi_verbs" in form:
        want_verbs = True
    if "fi_adjectives" in form:
        want_adjectives = True
    if "fi_pronouns" in form:
        want_pronouns = True
    if "vec_choice" in form:
        vec_choice = form["vec_choice"].value
        if "ftc_vecs" == vec_choice:
            vectorfile = "all_vec.bin"
        elif "s24_vecs" == vec_choice:
            vectorfile = "s24-vecs_no_pos.bin"
        elif "ftc_smaller_vecs" == vec_choice:
            vectorfile = "all_vec_top_half.bin"
        elif "s24_smaller_vecs" == vec_choice:
            vectorfile = "s24-vecs_no_pos_top_150_000.bin"
    if "lemmalist" in form:
        debuginfo += '<br>' + form["lemmalist"].value + '<br>\n'
        session_key.update(form["lemmalist"].value.encode("utf-8"))
        lemmalists = []
        this_lemmalist = ""
        for line in form["lemmalist"].value.split('\n'):
            line = line.strip()
            if line == '':
                if this_lemmalist != '':
                    lemmalists.append(this_lemmalist)
                    this_lemmalist = ''
                else:
                    continue
            else:
                this_lemmalist += line + '\n'
        if this_lemmalist != '':
            lemmalists.append(this_lemmalist)
        for i, lemmalist in enumerate(lemmalists):
            opts = {}
            opts["vectorfile"] = vectorfile
            lemmalist_script = make_script(lemmalist, opts)
            debuginfo += lemmalist_script + "<br>\n"
            lemmalist_pmatch_filename = wrkdir + "/tmp/" + session_key.hexdigest() + "_" + str(i + 1) + "_lemmalist.pmatch"
            process = Popen(["/usr/local/bin/hfst-pmatch2fst"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            out, err = process.communicate(input=bytes(lemmalist_script, "utf-8"))
            lemmalist_pmatch = open(lemmalist_pmatch_filename, "wb")
            lemmalist_pmatch.write(out)
            lemmalist_pmatch.close()
            matchers.append(lemmalist_pmatch_filename)
    inputstring = form["input"].value.strip().encode("utf-8")
    try:
        process = Popen([wrkdir + "/run-nertag"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        postag_out, err = process.communicate(input=inputstring)
    except Exception as ex:
        return "<p>Couldn't run postag! (Exception" + str(ex) + ")</p>\n"
    postag_out = postag_out.decode("utf-8")
    tokens = []
    for line in postag_out.split('\n'):
        if line == '':
            tokens.append(('', '', ['']))
        else:
            parts = line.split('\t')
            if len(parts) != 4:
                tokens.append((parts[0], '', ['']))
                debuginfo += "postag sent " + line + "<br>"
            else:
                tokens.append((parts[0], parts[1], [parts[2]]))
#    debuginfo += '<br>'.join(['\t'.join((parts[0], parts[1], parts[2][0])) for parts in tokens]).replace('\t', 'TAB') + "<br>"
        # for line in str(postag_out, "utf-8").split('\n'):
        #     line = line.strip()
        #     if line == '':
        #         tokens.append(('', '', ['']))
        #     else:
        #         parts = line.split('\t')
        #         if len(parts) != 3:
        #             tokens.append((line, '', ['']))
        #         else:
        #             tokens.append((parts[0], parts[1], [parts[2]]))
    debuginfo += "matchers: " + str(matchers) + "<br>\n"
    for matcher in matchers:
        process = Popen(["/usr/local/bin/hfst-pmatch", matcher], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        try:
            input_to_matcher = '\n'.join(['\t'.join((parts[0], parts[1], parts[2][0])) for parts in tokens])
            debuginfo += input_to_matcher + '<br>\n'
            matcher_out, matcher_err = process.communicate(input=bytes(input_to_matcher, "utf-8"))# (lambda x: x[1], tokens)), "utf-8"))
            matcher_out_lines = str(matcher_out, "utf-8").strip().split('\n')
            debuginfo += '<br>'.join(str(matcher_out, "utf-8").split('\n')) + "<br>\n"
        except Exception as ex:
            debuginfo += str(ex)
            return debuginfo

#            body += html_escape(str(matcher_out, "utf-8")) + " " + str(len(matcher_out_lines)) + " " + str(len(tokens))
#        debuginfo += "len(tokens) is " + str(len(tokens)) + ", len(matcher_out_lines) is " + str(len(matcher_out_lines)) + "<br>"
        for i in range(len(tokens)):
#            debuginfo += html_escape(matcher_out_lines[i]) + " ## " + str(tokens[i]) + "<br>"
            try:
                t = matcher_out_lines[i]
            except Exception as ex:
#                debuginfo += str(ex)
                continue
            while t.startswith('<') and '>' in t:
                tokens[i][2].append(t[:t.index('>') + 1])
                t = t[t.index('>') + 1:]
            while t.endswith('>') and '</' in t:
                tokens[i][2].append(t[t.rindex('</'):])
                t = t[:t.rindex('</')]
    debuginfo += xml_escape(str(tokens)) + "<br>\n"
   
    sentence = ""
    raw_sentence = ""
    paragraph = []
    raw_paragraph = []
    result = ""
    raw_result = ""
    guess_result = ""
    for i, token in enumerate(tokens):
        if token[0] == '':
            if sentence != '':
                result += wrap_in_tags(sentence, 'p')
                raw_result += raw_sentence + "\n\n"
#                paragraph.append(sentence)
                sentence = ""
                raw_sentence = ""
            # else:
            #     if len(paragraph) > 0:
            #         result += wrap_in_tags('<br>\n'.join(paragraph), 'p')
            #         paragraph = []
            continue
        if i > 0 and tokens[i - 1][0] != '' and token[0] not in space_eating_punct and i + 1 < len(tokens):
            sentence += " "
            raw_sentence += " "
        this_token = token[0]
        this_raw_token = token[0]
        if want_nouns and '[POS=NOUN]' in token[2][0]:
            this_token = wrap_noun(this_token)
        elif want_adjectives and '[POS=ADJECTIVE]' in token[2][0]:
            this_token = wrap_adjective(this_token)
        elif want_verbs and '[POS=VERB]' in token[2][0]:
            this_token = wrap_verb(this_token)
        elif want_pronouns and '[POS=PRONOUN]' in token[2][0]:
            this_token = wrap_pronoun(this_token)
        for tag in token[2][1:]:
            if tag == '<PMATCH_ATbold>':
                this_token = '<strong>' + this_token
            elif tag == '</PMATCH_ATbold>':
                this_token = this_token + '</strong>'
            elif tag.startswith('</'):
                this_token = this_token + xml_escape(tag)
                this_raw_token = this_raw_token + tag
            else:
                this_token = xml_escape(tag) + this_token
                this_raw_token = tag + this_raw_token
        sentence += this_token
        raw_sentence += this_raw_token
    if sentence != '':
        result += wrap_in_tags(sentence, 'p')
        raw_result += raw_sentence + "\n\n"
#        paragraph.append(sentence)
    # if len(paragraph) > 0:
    #     result += wrap_in_tags('<br>\n'.join(paragraph), 'p')
    result = wrap_in_tags(result, 'div', attribs = 'class="col-6"')
    return (wrap_in_tags(result + '\n' + guess_result, 'div', attribs = 'class="row"', oneline = False), raw_result)

def print_content():
    # logfile = open("lemmamatch_log.txt", "wb")
    # def log(s):
    #     logfile.write(s.encode("utf-8"))
    #     logfile.flush()
    time_start = time.time()
    form = cgi.FieldStorage()
    body = ""
    body += wrap_in_tags("lemmamatch demo", "h2")
    input_text = ""
    lemmalist_text = ""
    if "lemmalist" in form:
        lemmalist_text = form["lemmalist"].value
    if "input" in form:
        input_text = form["input"].value
    try:
        processed_form = process_form(form)
        if type(processed_form) == type(tuple()):
            processed_form, raw_text = processed_form[0], processed_form[1]
        else:
            raw_text = "Error in generating text"
    except Exception as ex:
        processed_form = "Couldn't process form: " + str(ex)
        raw_text = processed_form
    download_buttons = ''
    if processed_form != "":
        session_key = hashlib.md5(raw_text.encode("utf-8")).hexdigest()
        write_txt(raw_text, session_key)
        write_docx(raw_text, session_key, "lemmamatch output")
        download_buttons = '''
<div class="row">
  <div class="col-4">
    <a class="btn btn-info" role="button" href="{html_root}/kielipankki-tools/tmp/{filename}.txt", download="lemmamatch_result.txt">Download .txt result</a>
    <a class="btn btn-info" role="button" href="{html_root}/kielipankki-tools/tmp/{filename}.docx", download="lemmamatch_result.docx">Download .docx result</a>
  </div>
</div>
        '''.format(html_root = hostname, filename=session_key)
    body += '''
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle lead">Entering input</h6>
      <p class="card-text">
        The input text for processing is entered in the text box on the left.
      </p>
      <p class="card-text">
        The lemmas for matching are entered in the text box revealed by the "Add your own lemmas" button.
      </p>
      <p class="card-text">
        Word embeddings may be used with Like() and Unlike()...
      </p>
      <h6 class="card-subtitle lead">Understanding output</h6>
      <p class="card-text">
        The output text shows, in bold, tagged lemmas...
      </p>
    </div>
  </div>
</div>
<form method="post" action="/cgi-bin/{scriptname}">
  <div class="form-group">
    <div class="row">
      <div class="col-md-auto">
        <textarea name="input" type="text" rows="16" cols="42" class="form-control" id="inputted_text" placeholder="Text for matching goes here.">{textareatext}</textarea>
      </div>
      <div class="col-md-auto">
        <div class="form-check">
          <input class="form-check-input" type="checkbox" value="checked" id="fi_nouns_check" name="fi_nouns">
          <label class="form-check-label" for="fi_nouns_check" style="background-color:DodgerBlue;">Finnish nouns</label>
            </div>
        <div class="form-check">
          <input class="form-check-input" type="checkbox" value="checked" id="fi_verbs_check" name="fi_verbs">
          <label class="form-check-label" for="fi_verbs_check" style="background-color:MediumSeaGreen;">Finnish verbs</label>
        </div>
        <div class="form-check">
          <input class="form-check-input" type="checkbox" value="checked" id="fi_adjectives_check" name="fi_adjectives">
          <label class="form-check-label" for="fi_adjectives_check" style="background-color:Orange;">Finnish adjectives</label>
        </div>
        <div class="form-check">
          <input class="form-check-input" type="checkbox" value="checked" id="fi_pronouns_check" name="fi_pronouns">
          <label class="form-check-label" for="fi_pronouns_check" style="background-color:Violet;">Finnish pronouns</label>
        </div>
        <button class="btn btn-primary" type="button" data-toggle="collapse" data-target="#lemmalist_collapse" aria-expanded="false" aria-controls="collapseExample">Add your own lemmas</button>
        <div class="collapse" id="lemmalist_collapse">
          <textarea name="lemmalist" type="text" rows="10" cols="40" class="form-control" id="lemmalist_text" placeholder="Lemmalist goes here, ex:\n\nFoodName [This will become a tag]\nkokkelipiimä\nmämmi\n\n@bold [These will be bolded]\ntärkeä\nsana">{lemmalist}</textarea>
          <div class="radio"><label><input type="radio" name="vec_choice" value="ftc_vecs" checked="checked">Word embeddings from FTC newspapers, all lemmas</label></div>
          <div class="radio"><label><input type="radio" name="vec_choice" value="ftc_smaller_vecs">Word embeddings from FTC newspapers, top 50% lemmas</label></div>
          <div class="radio"><label><input type="radio" name="vec_choice" value="s24_vecs">Word embeddings from Suomi24 messages, all lemmas</label></div>
          <div class="radio"><label><input type="radio" name="vec_choice" value="s24_smaller_vecs">Word embeddings from Suomi24 messages, top 150K lemmas</label></div>
        </div>
      </div>
      <div class="col-2">
        <span class="label label-default">Clickable examples</span>
          <div class="border" onclick="populateTextField()">
            <p>LiittymäKännykkä<br>
              [Like(liittymä, kännykkä)^500]
            </p>
            <p>LiittymäEiKännykkä<br>
              [Unlike(liittymä, kännykkä)^500]
            </p>
            <p>Moottoritiellä on kaksi keskikaistan tai kaiteen toisistaan erottamaa...</p>
          </div>
      </div>
    </div>
    <div class="row">
        <div class="col">
          <button type="submit" class="btn btn-default">Submit</button>
        </div>
    </div>
  </div>
</form>
{result}
{downloadbutton}
<p><small>Page generated in {TIME_SPENT:.2f} seconds</small></p>
'''.format(result = processed_form, downloadbutton = download_buttons, lemmalist = lemmalist_text, textareatext = input_text, scriptname = os.path.basename(sys.argv[0]), TIME_SPENT = time.time() - time_start)
    sys.stdout.buffer.write(wrap_html(make_head("lemmamatch demo", scripts), wrap_in_tags(wrap_in_tags(body, "body", oneline = False), 'div', attribs='class="container pt-1"')).encode("utf-8"))

if __name__ == '__main__':
    print_content()

