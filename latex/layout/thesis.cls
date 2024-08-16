% This template was inspired by the TU Delft Report template and uses code from it [17-03-2022 v1.5 TU Delft Report Class] from Daan Zwaneveld ( see https://dzwaneveld.github.io/) . It also uses some parts of the code from the Uni Hamburg informatics VSIS workinggoup templates for thesis especially for the design of the titlepage (see https://vsis-www.informatik.uni-hamburg.de/vsis/teaching/templates?lang=en).


% This template is available under CC BY-NC 4.0. For more information, see https://creativecommons.org/licenses/by-nc/4.0/. No attribution is required in reports/theses created using this template.

%% -------------------------------------------------------------------
%%    Setting up the user-defined class
%% -------------------------------------------------------------------

%% this a user defined class Class is based on the default book class and some   %% extra options will be passed by this code
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{layout/thesis}[2022/12/24 v.0.1 SICSS thesis LaTeX class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[11pt,oneside]{book}

%% Main packages in the document --- Some are imported later in the class file
\RequirePackage{mathtools}  % Mathematical tools to use with amsmath
\RequirePackage{amssymb}    % Extended symbol collection
\RequirePackage{siunitx}    % Comprehensive (SI) units package

\RequirePackage{tabularx}   % Tabulars with adjustable-width columns
\RequirePackage{booktabs}   % Publication quality tables
\RequirePackage{longtable}  % Allow tables to flow over page boundaries
\RequirePackage{multirow}   % Create tabular cells spanning multiple rows

\RequirePackage{graphicx}   % Enhanced support for images
\RequirePackage{float}      % Improved interface for floating objects
\RequirePackage[labelfont=bf,justification=centering,footnotesize]{caption} % Captions
\RequirePackage{subcaption} % Support for sub-captions
\RequirePackage{pdfpages}   % Include PDF documents

\RequirePackage[pdfusetitle,hidelinks]{hyperref} % Extensive support for hypertext

% to use reference abbreviations remove noabbrev
% to make it work properly always use \label    after \caption
\RequirePackage[noabbrev,nameinlink,]{cleveref} % Intelligent cross-referencing


\RequirePackage{xcolor}     % Driver-independent color extensions
\RequirePackage{tikz}       % Create PostScript and PDF graphics
\RequirePackage{xspace}     % Define commands that appear not to eat spaces
\RequirePackage{microtype}  % Refinements towards typographical perfection

\RequirePackage{geometry}   % Customize document dimensions
\RequirePackage{titlesec}   % Select alternative section titles
\RequirePackage{titletoc}   % Alternative headings for toc
\RequirePackage{fancyhdr}   % Control of page headers and footers
\RequirePackage{enumitem}   % Control layout of itemize, enumerate, description
\RequirePackage{etoolbox}   % Toolbox of programming facilities
\RequirePackage{iftex}      % Adds if-else statements to support multiple compilers
\RequirePackage{datetime}   % Change format of \today


\RequirePackage{comment}    % for multiline comment blocks
\RequirePackage{chemformula} % for the chemical equations 
\RequirePackage{lipsum}     % for latin dummy text
\RequirePackage{siunitx}    % for SI units together with numbers
\RequirePackage{colortbl}   % for using colors in table environment
\RequirePackage{adjustbox}  % for including  graphics
\RequirePackage{wrapfig}    % to wrap figures
\RequirePackage{array}      % for tabular env  extends the options for column formats

%% Defining commands for the cover image, subtitle, subject, affiliation and table of authors
\newcommand*\subtitle[1]{\def\@subtitle{#1}}
\newcommand*\subject[1]{\def\@subject{#1}}
\newcommand*\affiliation[1]{\def\@affiliation{#1}}
\newcommand*\coverimage[1]{\def\@coverimage{#1}}
\newcommand*\covertable[1]{\def\@covertable{#1}}


%% Scaling the page margins  default is (.7)

%\geometry{a4paper,hscale=0.8,vscale=0.85}
\geometry{a4paper, width=15.5cm,height=22cm}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% I. Loading the fonts %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[T1]{fontenc}  % Selecting T1 font encoding
\RequirePackage{newpxtext}    % Palatino-like font...
\RequirePackage{newpxmath}    % ...with support for mathematics

%% Another font can be used as the large title style font if desired
\def\largetitlestyle{\normalfont}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% II. Adjusting the layout of the titles in the text and table of contents %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Formatting chapter titles and spacing
\titleformat{\chapter}[display]
{\flushright}
{\fontsize{80}{80}\selectfont\largetitlestyle\thechapter}
{0pt}
{\Huge}
\titlespacing*{\chapter}{0pt}{0pt}{2\baselineskip}


%% Formatting section titles and spacing
\titleformat{\section}
{\Large\bfseries}
{\thesection.}
{20pt}
{}

%% Formatting subsections titles and spacing
\titleformat{\subsection}
{\large\bfseries}
{\thesubsection.}
{5pt}
{}

%% Formatting subsubsections titles and spacing
\titleformat{\subsubsection}
{\bfseries}
{}
{0pt}
{}
\titlespacing*{\subsubsection}{0pt}{\bigskipamount}{0pt}

%% Reducing the vertical white space between chapters in the table of contents
\dottedcontents{chapter}[1.5em]{\vspace{0.5\baselineskip}\bfseries}{1.5em}{0pc}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% III. Adjusting the header and footer %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Change the header and footer of pages where chapters do not start (depends on twoside option)
\if@twoside
    \fancyhf{}
    \fancyhead[LE,RO]{\thepage}
    \fancyhead[RE]{\nouppercase{\leftmark}}
    \fancyhead[LO]{\nouppercase{\rightmark}}
    \RequirePackage{emptypage} % Remove header and footer on empty pages
\else
    \fancyhf{}
    \fancyhead[R]{\thepage}
    \fancyhead[L]{\nouppercase{\rightmark}}
\fi

\pagestyle{fancy} % Setting the package defaults and the additional changes as the style

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% IV. Setting up listings %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{listings} % Typeset source code listings

\lstset{
    basicstyle=\ttfamily\footnotesize,  % Style of the font that is used for the code
    backgroundcolor=\color{gray!10},    % Background color
    keywordstyle=\color{red!75!black},  % Keyword style
    stringstyle=\color{green!40!black}, % String style
    commentstyle=\color{blue!30!black}, % Comment style
    numbers=left,                       % Add line numbers on the left side
    numbersep=5pt,                      % Decrease distance between line numbers and code
    numberstyle=\tiny,                  % Line number style
    breaklines=true,                    % Line break automatically
}
