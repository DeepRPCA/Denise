"""Script to write .tex table with numerical comparison, given csv data file.

"""
import csv
import os.path

import pandas as pd

_PDF_TABLE_TMPL = r"""
\begin{table}[h]
\label{%(label)s}
\begin{center}
\resizebox{\columnwidth}{!}{
%(table)s
}
\end{center}
\caption{%(caption)s
}
\end{table}
"""


def _get_df():
  """Returns empty DataFrame with indices correctly initialized.
  """
  index_names = ['$s(S_0)$', 'Algo']
  algos = ["PCP", 'IALM', "FPCP", 'RPCA-GD', 'Denise']
  mi = pd.MultiIndex.from_product([
    [0.6, 0.7, 0.8, 0.9, 0.95],
    algos,
  ], names=index_names)
  result = pd.DataFrame(index=mi,
                        columns=['r_L', 's_S', 'rel_error_L', 'rel_error_S', 'time_(ms)'])
  return result


def _read_csv(csv_path):
  """Returns dataframe initialized using CSV data."""
  df = _get_df()
  with open(csv_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      line_index = (float(row['s_S0']), row['algo'])
      if line_index not in df.index:
        continue  # To filter out rows which we don't want.
      df.at[line_index, 'r_L'] = row['rankL']
      df.at[line_index, 's_S'] = row['sparsityS']
      df.at[line_index, 'rel_error_S'] = row['normSS']
      df.at[line_index, 'rel_error_L'] = row['normLL']
      df.at[line_index, 'time_(ms)'] = row['time']
  return df


def _write_table(csv_path, table_path):
  """Writes tex table to given path."""
  df = _read_csv(csv_path)
  print(df)
  print("Generating comparison table in %s" % table_path)
  caption = """Comparison between Denise and state of the art algorithms.
   For different given sparsity $s(S_0)$ of $S_0$, the output properties are the
	actual rank $r(L)$ of the returned matrix $L$, the sparsity $s(S)$ of the
	returned  matrix $S$ as well as the relative errors rel.error$(L)$ and
	rel.error$(S)$."""
  label = "table"
  pdf_table = _PDF_TABLE_TMPL % {
      "table": df.to_latex(
          na_rep="-", multirow=True, multicolumn=False,
          bold_rows=False,
          longtable=False,
          escape=False,
          header=["$r(L)$", "$s(S)$", "rel.error(L)", "rel.error(S)", "time (ms)"],
          column_format='| c c| c c |c  c |c|'),
      "caption": caption,
      "label": label,
  }

  # Color Denise cells in light-gray
  lines = pdf_table.split('\n')
  lines2 = []
  for line in lines:
    if ' Denise ' in line:
      line = ' & '.join(
          ['\\cellcolor{light-gray}' + cell if cell.strip() else cell
           for cell in line.split(' & ')])
    lines2.append(line)
  pdf_table = '\n'.join(lines2)

  with open(table_path, "w") as tablef:
    tablef.write(pdf_table)


def main():
  csv_path = "comparisons/comparison_overview_normal0.csv"
  table_path = "../latex_src/comparison_normal0.tex"
  _write_table(csv_path, table_path)

  csv_path = "comparisons/comparison_overview_student2.csv"
  table_path = "../latex_src/comparison_student2.tex"
  _write_table(csv_path, table_path)


if __name__ == "__main__":
  main()
