# Document fixtures

`sample.xls` is a synthetic BIFF8 workbook generated locally using xlwt 1.3.0.
It contains a Data sheet with the row `Alpha, empty, 42, true` and a
2024-01-02 date, plus a hidden sheet named Hidden containing Beta.
It contains no macros, external links, or user data. xlwt is only a fixture
generation tool and is not required to run the tests.

Other format fixtures are generated in temporary directories by
`test_document_files.py` using the runtime parser libraries. The XLSM fixture
uses the macro-enabled workbook content type without embedding VBA.
