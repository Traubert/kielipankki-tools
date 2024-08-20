#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import cgi
import time
import hashlib
import os
import re
import math
import cgitb
cgitb.enable()

from kpdemos import *

# populate_js = '''
# function processingTextField(demofile) {{
#   var xhttp = new XMLHttpRequest();
#   xhttp.onreadystatechange = function() {{
#     if (this.readyState == 4 && this.status == 200) {{
#       document.getElementById("inputted_text").value =
#         this.responseText;
#       }}
#     }};
#   xhttp.open("GET", "{HOSTNAME}/resources/" + demofile, true);
#   xhttp.send();
# }}
# '''.format(HOSTNAME=hostname)

def escape_js_string(s):
    return s.replace('"', '\\"').replace('\n', '\\n')

populate_js1 = '''
function populateTextField1() {
document.getElementById("inputted_text").value = "''' + escape_js_string(open('../resources/perustuslaki.txt', encoding = 'utf-8').read()) + '''";
}
'''

populate_js2 = '''
function populateTextField2() {
document.getElementById("inputted_text").value = "''' + escape_js_string(open('../resources/kekkonen.txt', encoding = 'utf-8').read()) + '''";
}
'''

colors = ('red', 'orange', 'yellow', 'green', 'blue', 'purple', 'grey', 'rust')
utils_js = '''
window.chartColors = {
	red: 'rgb(255, 99, 132)',
	orange: 'rgb(255, 159, 64)',
	yellow: 'rgb(255, 205, 86)',
	green: 'rgb(75, 192, 192)',
	blue: 'rgb(54, 162, 235)',
	purple: 'rgb(153, 102, 255)',
	grey: 'rgb(201, 203, 207)',
        rust: 'rgb(105,20,14)',
        brown: 'rgb(164,66,0)'
};

(function(global) {

	var Samples = global.Samples || (global.Samples = {});
	var Color = global.Color;

	Samples.utils = {
		numbers: function(config) {
			var cfg = config || {};
			var min = cfg.min || 0;
			var max = cfg.max || 1;
			var from = cfg.from || [];
			var count = cfg.count || 8;
			var decimals = cfg.decimals || 8;
			var continuity = cfg.continuity || 1;
			var dfactor = Math.pow(10, decimals) || 0;
			var data = [];
			var i, value;

			for (i = 0; i < count; ++i) {
				value = (from[i] || 0) + this.rand(min, max);
				if (this.rand() <= continuity) {
					data.push(Math.round(dfactor * value) / dfactor);
				} else {
					data.push(null);
				}
			}

			return data;
		},


		color: function(index) {
			return COLORS[index % COLORS.length];
		},

		transparentize: function(color, opacity) {
			var alpha = opacity === undefined ? 0.5 : 1 - opacity;
			return Color(color).alpha(alpha).rgbString();
		}
	};



}(this));
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

column_names = ["Nouns", "Verbs", "Adjectives", "NER entities"]

n_words = 20
max_gold_lemmas = 100000

def process_input(inputval):
    result = ""
    try:
        process = Popen([wrkdir + "/run-nertag"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = process.communicate(input=inputval)
        session_key = hashlib.md5(out).hexdigest()
        out_rows = tsv2rows(out.decode("utf-8"))
#        return(str(inputval))
        surfaces = extract_column(out_rows, 0)
        lemmas = extract_column(out_rows, 1)
        tags = extract_column(out_rows, 2)
        nertags = extract_column(out_rows, 3)
        n_lemmas = len(lemmas)
        ner_dict = {}
        this_nertag = []
        noundict = {}
        verbdict = {}
        adjectivedict = {}
        tokens = []
        for i in range(n_lemmas):
            if nertags[i] == '':
                if len(this_nertag) > 0:
                    this_nertag.append(surfaces[i])
                else:
                    if '=NOUN' in tags[i]:
                        noundict[lemmas[i]] = noundict.get(lemmas[i], 0) + 1
                    elif '=VERB' in tags[i]:
                        verbdict[lemmas[i]] = verbdict.get(lemmas[i], 0) + 1
                    elif '=ADJECTIVE' in tags[i]:
                        adjectivedict[lemmas[i]] = adjectivedict.get(lemmas[i], 0) + 1
                    tokens.append(lemmas[i])
                continue
            if nertags[i].startswith('</') or nertags[i].endswith('/>'):
                lemma = lemmas[i]
                surface = surfaces[i]
                if 'PROPER' in tags[i]:
                    if surface.isupper():
                        lemma = lemma.upper()
                    elif surface.istitle():
                        lemma = lemma.title()
                this_nertag.append(lemma)
#                ners.append([this_nertag, nertags[i][2:-1], i])
                tagged = ' '.join(this_nertag)
                ner_dict[tagged] = ner_dict.get(tagged, 0) + 1
                this_nertag = []
                tokens.append(tagged)
            else:
                if 'CASE=GEN' in tags[i] or 'CASE=NOM' in tags[i]:
                    this_nertag.append(surfaces[i])
                else:
                    this_nertag.append(lemmas[i])
                tokens.append('')
        gold_worddict = dict(line.strip().split(" ") for line in open("lemmafreq.txt", encoding = "utf-8").readlines()[:max_gold_lemmas])
        nounfreqlist = []
        verbfreqlist = []
        adjectivefreqlist = []
        for worddict, freqlist in ((noundict, nounfreqlist),
                                   (verbdict, verbfreqlist),
                                   (adjectivedict, adjectivefreqlist)):
            for key, value in worddict.items():
                if value == 1:
                    continue
                freqlist.append((key, -1*math.log(float(value)/n_lemmas)))
            freqlist.sort(key = lambda pair: pair[1])
        nouns_in_gold_scored = []
        verbs_in_gold_scored = []
        adjectives_in_gold_scored = []
        words_not_in_gold = []
        for words_in_gold_scored, freqlist in ((nouns_in_gold_scored, nounfreqlist),
                                               (verbs_in_gold_scored, verbfreqlist),
                                               (adjectives_in_gold_scored, adjectivefreqlist)):
            for word in freqlist:
                if word[0] in gold_worddict:
                    words_in_gold_scored.append((word[0], word[1] / float(gold_worddict[word[0]])))
                else:
                    words_not_in_gold.append(word[0])
            words_in_gold_scored.sort(key = second)
        nouns_in_gold = list(map(first, nouns_in_gold_scored))[:n_words]
        verbs_in_gold = list(map(first, verbs_in_gold_scored))[:n_words]
        adjectives_in_gold = list(map(first, adjectives_in_gold_scored))[:n_words]
        ners_ = [pair[0] for pair in sorted(ner_dict.items(), reverse = True, key = lambda x: x[1])[:n_words]]
        words_not_in_gold = words_not_in_gold[:n_words]
        ners_.sort(key = lambda x: ner_dict[x], reverse = True)
        indexed_nouns_in_gold = list(zip(nouns_in_gold, range(len(nouns_in_gold))))
        indexed_verbs_in_gold = list(zip(verbs_in_gold, map(lambda x: x + len(nouns_in_gold), range(len(verbs_in_gold)))))
        indexed_adjectives_in_gold = list(zip(adjectives_in_gold, map(lambda x: x + len(nouns_in_gold) + len(verbs_in_gold), range(len(verbs_in_gold)))))
        indexed_ners_ = list(zip(ners_, map(lambda x: x + len(nouns_in_gold) + len(verbs_in_gold) + len(adjectives_in_gold), range(len(ners_)))))
        indexed_words_not_in_gold = list(zip(words_not_in_gold, map(lambda x: x + len(nouns_in_gold) + len(verbs_in_gold) + len(adjectives_in_gold) + len(ners_), range(len(words_not_in_gold)))))
#        result_rows = cols2rows(words_in_gold, ners_, words_not_in_gold)
        indexed_result_rows = cols2rows((indexed_nouns_in_gold, indexed_verbs_in_gold, indexed_adjectives_in_gold, indexed_ners_), pad_with = ('', None))
#            write_excel([column_names] + result_rows, session_key, "Output from lemmarank")
#            write_tsv(make_tsv([column_names] + result_rows), session_key)
    except Exception as ex:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        return "<p>Got exception " + str(ex) + " in file " + str(fname) + " line " + str(exc_tb.tb_lineno) + "</p>\n"

        # result += '''
        # <div class="row">
        # <div class="col-4">
        # </div>
        # </div>
        # '''#.format(html_root = hostname, filename = session_key)

        # <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.tsv" download="ner_tagged.tsv" role="button">Download TSV</a>
        # <a class="btn btn-info" href="{html_root}/kielipankki-tools/tmp/{filename}.xlsx" download="ner_tagged.xlsx" role="button">Download Excel spreadsheet</a>

    noun_datasets = []
    verb_datasets = []
    adjective_datasets = []
    ner_datasets = []
    other_datasets = []
    all_datasets = []
    for j in range(5):
        for i in range(n_words):
            wordlist = (nouns_in_gold, verbs_in_gold, adjectives_in_gold, ners_, words_not_in_gold)[j]
            dataset = (noun_datasets, verb_datasets, adjective_datasets, ner_datasets, other_datasets)[j]
            if i >= len(wordlist):
                continue
            thisdata = "{ label: '" + wordlist[i] + "',\n"
            thisdata += "backgroundColor: window.chartColors.{},\n".format(colors[len(all_datasets) % len(colors)])
            thisdata += "borderColor: window.chartColors.{},\n".format(colors[len(all_datasets) % len(colors)])
            thisdata += "data: "
            counts = []
            for section in range(20):
                counts.append(len(list(filter(lambda x: x == wordlist[i], tokens[int(section*0.05*len(tokens)):int((section+1)*0.05*len(tokens)) - 1]))))
            thisdata += str(counts) + ",\n"
            thisdata += "fill: false\n}"
            dataset.append(thisdata)
            all_datasets.append(thisdata)

    def my_make_table(rows, header = [], tdattribs = ""):
        retval = ''.join(list(map(lambda x: wrap_in_tags(x, 'th'), header)))
        for row in rows:
            this_row = ""
            for item in row:
#                if (item[1] != None):
                this_row += wrap_in_tags(item[0], "td", attribs=tdattribs.format(CELL=str(item[1])))
#                else:
#                    this_row += wrap_in_tags(item[0], "td")
            retval += wrap_in_tags(this_row, "tr", oneline = False)
        return wrap_in_tags(retval, "table", oneline = False, attribs = 'class = table table-bordered')

        
    table_code = wrap_in_tags(my_make_table(indexed_result_rows, header = column_names, tdattribs="onclick='show(\"{CELL}\")'" + ' id="cell_{CELL}"') + "\n", "div", attribs='class="col-md-auto"', oneline = False)
    canvas_code = '''<div class="col-md-auto" width="100%">
    <canvas id="lemmas_chart" width="720px"></canvas>
    </div>
    '''

    result += wrap_in_tags(canvas_code, 'div', attribs='class="row"', oneline = False)
    result += wrap_in_tags(table_code, 'div', attribs='class="row"', oneline = False)

    result += '''
    <script>
    alldatasets = [
    '''
    result += ', '.join(all_datasets) + ']\n\n'
    result += '''
    used_datasets = new Set();
    var config_lemmas = {
    type: 'line',
    data: {
    labels: ['5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%', '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%', '100%'],
    datasets: [],
    options: {
    responsive: true,
    title: {
    display: true,
    text: 'Appearances in text'
    },
    tooltips: {
    mode: 'index',
    intersect: false,
    },
    hover: {
    mode: 'nearest',
    intersect: true
    },
    scales: {
    xAxes: [{
    display: true,
    scaleLabel: {
    display: true,
    labelString: 'Section of text'
    }
    }],
          yAxes: [{
            display: true,
            scaleLabel: {
              display: true,
              labelString: 'Value'
            }
          }]
        }
      }
    }
  }
    
    show = function(index) {
        //alert(index)
        idx = parseInt(index);
        if (used_datasets.has(idx)) {
            used_datasets.delete(idx);
        } else {
            used_datasets.add(idx);
        }
        document.getElementById("cell_" + index).classList.toggle("bg-primary");
        new_datasets = [];
        used_datasets.forEach(function(i) {new_datasets.push(alldatasets[i])});
        config_lemmas["data"]["datasets"] = new_datasets;
        window.myLine.update();
        };


    window.onload = function() {
var ctx = document.getElementById('lemmas_chart').getContext('2d');
window.myLine = new Chart(ctx, config_lemmas);
    show(0); show(1); show(2);
    };

</script>
'''
    return result

def print_content():
    time_start = time.time()
    clean_tempfiles()
    form = cgi.FieldStorage()
    result = ""
    inputval = ""
    if "input" in form:
        inputval = form["input"].value.encode("utf-8")
    if "file" in form and form["file"].filename != "":
#        pass
#        result += "Got file " + str(form["file"].filename)
        inputval = text_from_file(form["file"])
    if inputval != "":
        result += str(process_input(inputval))

    body = wrap_in_tags("lemmarank demo", "h2")
    body += '''
<h6>Rank representative lemmas and named entities from running text</h6>
<a href="#help" data-toggle="collapse">Show help</a>
<div class="collapse" id="help">
  <div class="card" style="width: 40rem;">
    <div class="card-body">
      <h4 class="card-title"><u>Help</u></h4>
      <h6 class="card-subtitle lead">Entering input</h6>
      <p class="card-text">
        You have a choice between three options: enter text in the text box, choose one of the demo text options, or upload a file. A variety of file formats are supported: plain utf-8 text (.txt), and unless the formatting is especially convoluted, .pdf, .doc, .docx, .csv, .epub, .html, .odt, .rtf and .xls files.
      </p>
      <h6 class="card-subtitle lead">Understanding output</h6>
      <p class="card-text">
        The output is represented in two ways: a table and a dynamic graph.
      </p>
      <p class="card-text">
        The table has four columns. The first three show the base forms of nouns, verbs, and adjectives that are especially frequent in this text compared to a reference corpus, and the fourth shows named entities (NERs; people, places, organisations, time expressions). The named entities are ranked by absolute frequency, ie. not relative to a reference corpus.
      </p>
      <p class="card-text">
        The graph starts out by plotting the frequency of the three most representative lemmas. They are highlighted in blue in the table. By clicking on cells in the table, you can toggle the display of each one on and off. The text is divided into 20 sections, each comprising 5% of the text, and the plot shows a count of each cell over a particular 5% section.
      </p>
    </div>
  </div>
</div>
<form method="post" action="/cgi-bin/{scriptname}" enctype="multipart/form-data">
  <div class="form-group">
    <div class="row">
      <div class="col-md-auto">
        <textarea name="input" type="text" rows="10" cols="60" class="form-control" id="inputted_text" placeholder="Enter input text here."></textarea>
      </div>
    </div>
    <div class="row">
      <div class="col-md-auto">
        <div class="dropdown" name="demotext">
          <button type="button" class="btn btn-default dropdown-toggle" data-toggle="dropdown"><b>Or</b> select demo text</button>
          <div class="dropdown-menu">
            <div class="dropdown-item" onclick="populateTextField1()">Constitution of the Finnish republic</div>
            <div class="dropdown-item" onclick="populateTextField2()")">Urho Kekkonen Wikipedia page</div>
          </div>
        </div>
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
{content}
<p><small>Page generated in {TIME_SPENT:.2f} seconds</small></p>
'''.format(scriptname = os.path.basename(sys.argv[0]), content = result, TIME_SPENT = time.time() - time_start)
    sys.stdout.buffer.write(wrap_html(make_head(title = 'lemmarank demo', scripts = (populate_js1, populate_js2, utils_js)), wrap_in_tags(body, 'div', attribs='class="container pt-1"')).encode("utf-8"))

if __name__ == '__main__':
    print_content()
