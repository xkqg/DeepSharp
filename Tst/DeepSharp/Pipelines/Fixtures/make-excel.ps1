# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

# Writes the workbooks the Excel reader's tests read, with Microsoft Excel itself, through its COM interface: run once on
# a Windows machine with Excel, from this folder, with Windows PowerShell. What it writes is committed beside it.
#
#   titanic.xlsx, titanic.xls   Samples/data/titanic.csv, every cell as the CSV holds it. The columns whose every cell
#                               is a whole number (survived, pclass, sibsp, parch) are numbers and those that hold True
#                               or False (adult_male, alone) are true or false, since the reader hands those back as the
#                               CSV writes them; every other column is text, as the CSV spells it — 22.0, 71.2833 —
#                               since a number cell would be handed back in its own spelling, 22. An empty CSV cell is
#                               an empty cell.
#   people.xlsx, people.xls,   two sheets, 'people' and 'pets': words with letters beyond ASCII, a number with a
#   people.xlsb                 fraction, a date typed as a date, true and false, a number written as text, an empty
#                               cell and a formula that divides by nought.

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$csv = Join-Path $here '..\..\..\..\Samples\data\titanic.csv'

$xlOpenXMLWorkbook = 51
$xlExcel12 = 50
$xlExcel8 = 56

$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false

try
{
    # The passenger list.
    $lines = Get-Content -LiteralPath $csv
    $names = $lines[0].Split(',')
    $rows = $lines.Count
    $numbers = @('survived', 'pclass', 'sibsp', 'parch')
    $truths = @('adult_male', 'alone')
    $cells = New-Object 'object[,]' $rows, $names.Count

    for ($column = 0; $column -lt $names.Count; $column++)
    {
        $cells[0, $column] = $names[$column]
    }

    for ($row = 1; $row -lt $rows; $row++)
    {
        $values = $lines[$row].Split(',')

        for ($column = 0; $column -lt $names.Count; $column++)
        {
            $value = $values[$column]

            if ($value -eq '')
            {
                continue
            }

            if ($numbers -contains $names[$column])
            {
                $cells[$row, $column] = [double]::Parse($value, [Globalization.CultureInfo]::InvariantCulture)
            }
            elseif ($truths -contains $names[$column])
            {
                $cells[$row, $column] = [bool]::Parse($value)
            }
            else
            {
                $cells[$row, $column] = $value
            }
        }
    }

    $book = $excel.Workbooks.Add()
    $sheet = $book.Worksheets.Item(1)
    $sheet.Name = 'titanic'

    # Text columns are formatted as text before anything is written, so Excel keeps 22.0 as the words it is.
    for ($column = 0; $column -lt $names.Count; $column++)
    {
        if (($numbers -notcontains $names[$column]) -and ($truths -notcontains $names[$column]))
        {
            $sheet.Columns.Item($column + 1).NumberFormat = '@'
        }
    }

    $sheet.Range($sheet.Cells.Item(1, 1), $sheet.Cells.Item($rows, $names.Count)).Value2 = $cells
    $book.RemovePersonalInformation = $true
    $book.SaveAs((Join-Path $here 'titanic.xlsx'), $xlOpenXMLWorkbook)
    $book.SaveAs((Join-Path $here 'titanic.xls'), $xlExcel8)
    $book.Close($false)

    # A small workbook of every kind of cell, on two sheets.
    $book = $excel.Workbooks.Add()

    while ($book.Worksheets.Count -lt 2)
    {
        [void]$book.Worksheets.Add([Type]::Missing, $book.Worksheets.Item($book.Worksheets.Count))
    }

    $people = $book.Worksheets.Item(1)
    $people.Name = 'people'
    $people.Columns.Item(5).NumberFormat = '@'
    $people.Cells.Item(1, 1).Value2 = 'name'
    $people.Cells.Item(1, 2).Value2 = 'age'
    $people.Cells.Item(1, 3).Value2 = 'born'
    $people.Cells.Item(1, 4).Value2 = 'adult'
    $people.Cells.Item(1, 5).Value2 = 'note'
    # Written by its code points, since Windows PowerShell reads a script without a byte-order mark in the machine's code page.
    $people.Cells.Item(2, 1).Value2 = 'Zo' + [char]0x00EB + ' Caf' + [char]0x00E9
    $people.Cells.Item(2, 2).Value2 = 22.5
    $people.Cells.Item(2, 3).Value2 = 42053
    $people.Cells.Item(2, 3).NumberFormat = 'yyyy-mm-dd'
    $people.Cells.Item(2, 4).Value2 = $true
    $people.Cells.Item(2, 5).Value2 = '22.0'
    $people.Cells.Item(3, 1).Value2 = 'Bob'
    $people.Cells.Item(3, 3).Formula = '=1/0'
    $people.Cells.Item(3, 4).Value2 = $false

    $pets = $book.Worksheets.Item(2)
    $pets.Name = 'pets'
    $pets.Cells.Item(1, 1).Value2 = 'kind'
    $pets.Cells.Item(1, 2).Value2 = 'legs'
    $pets.Cells.Item(2, 1).Value2 = 'cat'
    $pets.Cells.Item(2, 2).Value2 = 4
    $pets.Cells.Item(3, 1).Value2 = 'bird'
    $pets.Cells.Item(3, 2).Value2 = 2

    $book.RemovePersonalInformation = $true
    $book.SaveAs((Join-Path $here 'people.xlsx'), $xlOpenXMLWorkbook)
    $book.SaveAs((Join-Path $here 'people.xls'), $xlExcel8)
    $book.SaveAs((Join-Path $here 'people.xlsb'), $xlExcel12)
    $book.Close($false)
}
finally
{
    $excel.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
}
