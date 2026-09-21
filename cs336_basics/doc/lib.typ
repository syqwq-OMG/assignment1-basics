#let report(body, tit: "Title", name: "Yuquan Sun", student_id: "10234900421", date: "2026.9.22") = {
  set text(font: ("New Computer Modern", "Source Han Serif SC"))
  set document(title: tit)

  set heading(numbering: "1.1")
  set page(numbering: "1")

  show raw:set text(font:"Fira Code")

  align(center, title())
  align(center, text(size: 14pt)[#name #student_id\ #date])
  body
}
