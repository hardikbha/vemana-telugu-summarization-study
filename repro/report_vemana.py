"""Build a reviewable report of the actual Vemana-Base trial."""
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
TOPICS = {"water": ["water"], "pension": ["pension"], "roads": ["road", "pothole"],
          "electricity": ["electric", "power", "voltage"], "sanitation": ["garbage", "waste", "sewage", "drain"],
          "healthcare": ["medicine", "medication", "health", "medical"], "ration": ["ration", "food"],
          "land": ["land", "record"], "education": ["school", "toilet"], "urgent_wires": ["wire"]}


def main():
    folder = ROOT / "vemana_evaluation"
    result = json.loads((folder / "results.json").read_text())
    dataset = json.loads((ROOT / "benchmark_100/dataset.json").read_text())
    expected = {row["id"]: row for row in dataset["notes"]}
    rows = result["long_note_runs"]
    assert len(rows) == 100 and {row["id"] for row in rows} == set(expected)
    assert all("batch_size" in row for row in rows)
    assert all(not row["source_truncated"] for row in rows)
    assert all(row["input_tokens"] + max(0, row["max_new_tokens"]) <= 1024 for row in rows)
    copied = sum("garbage has not been collected" in row["output"].lower() for row in rows)
    topic_present = sum(any(term in row["output"].lower() for term in TOPICS[expected[row["id"]]["category"]]) for row in rows)
    analysis = {"model": result["model"], "license": result["license"], "parameters": result["parameters"],
                "model_revision": result["revision"], "actual_model_bytes": sum(p.stat().st_size for p in (ROOT/"models/Vemana-Base").glob('*') if p.is_file()),
                "long_notes_attempted": 100, "accepted_basic_checks": result["accepted_basic_checks"],
                "rejected": result["rejected_outputs"], "context_failures": result["context_budget_failures"],
                "copied_garbage_example_phrase": copied, "source_topic_keyword_present": topic_present,
                "diagnostic_limits": "Basic checks and keyword counts are not factual-accuracy scores. Copying the prompt example can create false topical matches.",
                "peak_process_ram_gib": result["peak_process_memory_gib"], "runtime_seconds": result["elapsed_seconds"],
                "device": result["device"], "trained_context_limit": 1024, "network_attempts": result["network_attempts"],
                "recommendation": "Vemana-Base is not suitable as a drop-in Telugu-to-English summarizer in this trial. It is an Apache-2.0 base for further research/adaptation, not a validated summarization model.",
                "comparison_caveat": "Different architecture, tokenizer, prompt format, precision and output budget from Qwen; no controlled model ranking is claimed."}
    (folder / "analysis.json").write_text(json.dumps(analysis, indent=2)+'\n')
    with (folder / "all_outputs.csv").open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','source_category','status','input_tokens','generated_tokens','seconds','output'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row.get(k,'') for k in writer.fieldnames},'source_category':expected[row['id']]['category']})
    styles = getSampleStyleSheet()
    normal = ParagraphStyle('Review',parent=styles['BodyText'],fontSize=9,leading=12,textColor=colors.HexColor('#18324a'))
    small = ParagraphStyle('Small',parent=normal,fontSize=8,leading=10)
    title = ParagraphStyle('ReportTitle',parent=styles['Title'],fontSize=19,leading=23,textColor=colors.HexColor('#133758'))
    flow = [Paragraph('Vemana-Base: Telugu → English evaluation',title),
            Paragraph('Actual offline inference | 3 short pilot notes + the same 100 long synthetic Telugu notes',small),Spacer(1,10),
            Paragraph('Model and license',styles['Heading2']),
            Paragraph(f"udgamlabs/Vemana-Base; {result['parameters']:,} parameters; Apache 2.0. The model card identifies it as a Telugu base model that continues text rather than following instructions. Its trained context is 1,024 tokens. No fine-tuning or model substitution was performed.",normal),
            Spacer(1,8),Paragraph('Observed results',styles['Heading2'])]
    table = Table([['Measurement','Result'],['Long notes attempted','100 (302–317 words each)'],
                   ['Passed basic English/completion checks',str(result['accepted_basic_checks'])+'/100 — not accuracy'],
                   ['Rejected outputs',str(result['rejected_outputs'])],['Context-budget failures',str(result['context_budget_failures'])],
                   ['Copied garbage-example phrase',str(copied)+'/100'],
                   ['Source-topic keyword present',str(topic_present)+'/100 — weak diagnostic only'],
                   ['100-note run / peak process RAM',f"{result['elapsed_seconds']/60:.2f} min / {result['peak_process_memory_gib']:.2f} GiB"],
                   ['Execution','CPU FP32; long-note batches of '+str(result['long_note_batch_size'])]],
                  colWidths=[270,230])
    table.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),8),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                              ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf4f8')),
                              ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d7e2ea')),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    flow.extend([table,Spacer(1,8),Paragraph('Examples from the short pilots — model output, not corrected',styles['Heading2'])])
    examples = [('pilot_1_direct','Water outage → direct prompt'),('pilot_1_fewshot','Water outage → one-example prompt'),
                ('pilot_2_fewshot','Pension delay → one-example prompt')]
    pilots={row['id']:row for row in result['pilot_runs']}
    for id,label in examples:
        output=pilots[id]['output']
        snippet=' '.join(output.split()[:24])
        if len(output.split())>24:snippet+=' … [excerpt]'
        flow.extend([Paragraph(escape(label)+':',small),Paragraph(escape(snippet),normal),Spacer(1,5)])
    flow.extend([Paragraph('Assessment and limits',styles['Heading2']),
                 Paragraph('The direct water prompt produced an unrelated crime story; the few-shot prompt copied its garbage example and invented details. English-looking output is not evidence of source-faithful summarization. The model loads and generates Telugu, but this trial does not support using it for the requested summarizer. Task-specific supervised adaptation would need a separate evaluation.',normal),
                 Spacer(1,5),Paragraph('All source text was retained without truncation; plain completions were used because no chat template is supplied. Outputs and hashes are in results.json/all_outputs.csv. The synthetic dataset and prompt differences prevent a controlled accuracy comparison with Qwen. No GPU or AWS instance was used for this trial.',small),
                 Paragraph('Model card/license: huggingface.co/udgamlabs/Vemana-Base',small)])
    pdf=folder/'Vemana_Base_Evaluation.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=35).build(flow)
    assert len(PdfReader(pdf).pages)==1
    doc=Document();doc.core_properties.title='Vemana-Base evaluation';doc.styles['Normal'].font.size=Pt(10)
    doc.styles['Normal'].font.name='Nirmala UI'
    doc.add_heading('Vemana-Base: Telugu-to-English summarization test',0)
    doc.add_paragraph(analysis['recommendation'])
    doc.add_paragraph('Open license: Apache 2.0. Model card: https://huggingface.co/udgamlabs/Vemana-Base')
    doc.add_heading('Measurements',1)
    for k in ['parameters','long_notes_attempted','accepted_basic_checks','rejected','context_failures','copied_garbage_example_phrase','source_topic_keyword_present','peak_process_ram_gib','runtime_seconds']:
        doc.add_paragraph(k.replace('_',' ')+': '+str(analysis[k]))
    doc.add_paragraph(analysis['diagnostic_limits']);doc.add_paragraph(analysis['comparison_caveat'])
    doc.add_heading('All short pilot outputs, unedited',1)
    for row in result['pilot_runs']:
        doc.add_heading(row['id'],2);doc.add_paragraph('Telugu source: '+row['source']);doc.add_paragraph('Model output: '+row['output'])
        doc.add_paragraph('Status: '+row['status']+'; checks: '+', '.join(row.get('errors',[])))
    doc.add_heading('Telugu continuation sanity check',1);doc.add_paragraph(result['telugu_continuation'])
    doc.save(folder/'Vemana_Base_Evaluation.docx')
    archive=ROOT/'vemana_evaluation_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in folder.rglob('*'):
            if path.is_file():z.write(path,'vemana_evaluation/'+str(path.relative_to(folder)))
        for name in ['LICENSE','NOTICE','README.md','model_manifest.json']:
            z.write(ROOT/'models/Vemana-Base'/name,'model_metadata/'+name)
        z.write(ROOT/'benchmark_100/dataset.json','source_dataset/dataset.json')
    print(json.dumps({**{k:v for k,v in analysis.items() if k not in ['network_attempts']},'pdf':str(pdf),'results_zip':str(archive)},indent=2))


if __name__=='__main__':
    main()
