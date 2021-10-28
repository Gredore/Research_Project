The CIFs in this folder contain derived data from the 
Non-disordered MOF subset of the Cambridge Structural Database v542

This work is licensed under the Creative Commons 
Attribution-NonCommercial-ShareAlike 4.0 International License.
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

The CIF files were generated via a Python script on a Windows system
and therefore have Windows End of Line (EOL) characters (CR and LF or '\r\n')

The original CSD data for 3D frameworks containing over 10 % void space
has been converted to spacegroup P1 and all non framework solvent molecules
(i.e. all uncharged molecules) have been removed. The probe radius of 1.32A
for the void calculation is based on a He atom with a 'contact' interaction
where the whole probe must be accomodated. Any frameworks with missing 
hydrogen atom positions have had these atomic coordinates added based on
standard bond lengths and geometries. In a small number of cases, hydrogen
addition to atoms on a special position can result in an unreliable chemical
representation; for these structures the hydrogen is not added and the CIF
labelled accordingly. These frameworks are likely to contain oxygen atoms
with incomplete coordination.

The CIF headers contain information regarding the original dataset including
CSD Refcode, crystal system and void space.

The CIF filenames have a consistent syntax consisting of the CSD refcode in
lowercase followed by '_P1' to highlight the converted spacegroup symmetry.
For charged frameworks the filename includes an additional '_charged' suffix,
and frameworks that have had hydrogen added include '_H' in the filename.

The folder also contains four reference spreadsheet files (.csv format)

The 'Framework details.csv' file lists all the frameworks, giving details of
the original CSD Refcode, CIF filename, original crystal system and percentage
void space (details of the void calculation above). The spreadsheet also
contains columns marking various properties that can be filtered to create
smaller sets of data. These columns record if the framework was in a 'chiral'
(Sohncke) space group, if the framework is charged and if hydrogen atom 
positions were added. As detailed above, in a small number of cases hydrogen
addition can result in an unreliable chemical representation, these situations
are recorded in the final column of the spreadsheet.

The remaining three spreadsheets report the three subcategories of CIFs
produced by the current process and contain the CSD Refcode of the original
data and the filename of the generated CIF. These are intended for reference 
so that a high-throughput analysis can restrict to (or exclude) these
structures.


