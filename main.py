from system import GraphRAGSystem

def main():
    """Example usage of the simplified Graph RAG system"""
    
    # Sample documents
    documents = [
        """
        Proposal for Curriculum Change 
to be approved by Academic Programs Committee 
PROPOSAL IDENTIFICATION 
Title of proposal: Minor in Crime, Law, and Justice Studies 
Degree(s): Bachelor of Arts 
Field(s) of Specialization: Sociology 
Level(s) of Concentration: Minor 
Option@): NIA 
Degree College: Arts and Science 
Department: Sociology 
Home College: Arts and Science 
Contact person(s) (name, telephone, fax, e-mail): Patricia Monture, Chair, 
Undergraduate Studies, Department of Sociology, Tel: 966-6959; Email: 
trisha.monture@usask.ca 
Date: 
Proposed date of implementation: September 2007 
RATIONALE 
This proposal is based on the student demand for criminology courses. It also 
reflects the department's desire to provide non-sociology students who complete 
the core courses in sociology in the area of crime, law and justice studies with an 
acknowledgement of their accomplishment. 
The minor in Crime, Law and Justice Studies, compliments the programs offered 
in Native Studies, Political Studies, Psychology and Public Administration as well 
as being of interest to students in other disciplines. 
With the development of the minor in Crime, Law and Justice Studies, 
department members felt strongly that students in our program should be 
respected and have the same opportunity for acknowledgement in this 
specialization as non-sociology students. 
The Sociology department has for the last few years been engaged in revisions 
of their program so it more accurately reflects what the department does and the 
opportunities sociology offers to students. As part of this department plan, the 
department desires to find ways to reflect the "streams" or specializations, of 
which criminology is one, available to students in our department. Additionally, 
these objectives reflect the guidance the department received in the last 
departmental review. We believe this strengthens the opportunities available in 
the college and provides another opportunity for cross-disciplinary study. 
As other universities have criminology or sociology of law degree programs (U of 
TI Ottawa, Carlton, and SFU) this allows the department to be more competitive 
nationally. 
DESCRIPTION OF PROGRAM CHARACTERISTICS 
Calendar Description: 
The minor in Crime, Law and Justice Studies may be completed in conjunction 
with any Three-Year, Four-Year or Honours degree in the College of Arts and 
Science. 
Non-Sociology Majors 
Students must complete 21 credit units Sociology: - SOC 212.3, 232.3, 233.3, 234.3; SOC 214.3 or 219.3; - 6 credit units selected from SOC 31 1.3, 312.3, 329.3 334.3 or 341.3.* 
Sociology Majors (B.A. Four-year only) 
In addition to the requirements for the B.A. Four-year Sociology, students must 
complete 18 credit units Sociology: - 15 credit units selected from SOC 212.3, 234.3; one of SOC 214.3 or 219.3; 
any two of SOC 31 1.3, 312.3, 329.3, 334.3 or 341.3 - 3 credit units selected from SOC 41 5.3,419.3 or 439.3; or a class from list 
provided above not already completed as a minor requirement.* 
Note: Students may not count the same courses towards the requirements for 
both a major and a minor subject area and a maximum of 60 credit units are 
allowed in one subject for the B.A. Four-year. 
*In addition to the courses required for the Minor in Crime, Law and Justice 
Studies, students are encouraged to complete SOC 203.3, 205.3, or 206.3. 
RESOURCES 
New program does not require additional resources. As well, additional library, 
laboratory, information technology and equipment resources are not needed. 
RELATIONSHIPS AND IMPACT OF IMPLEMENTATION 
The department does not believe this will impact on any other departments or 
programs. Consultation with the College of Law was not required given the 
nature of the training that goes on in law schools (preparing people for the 
practice of law); there really isn't any overlap. The minor is a sociological 
approach to law which is vastly different. 
BUDGET 
Teaching and other course expenses will be accommodated within departmental 
budget. 
COLLEGE STATEMENT 
The proposed Minor in Crime, Law, and Justice Studies has been approved by 
the Arts & Science Programs Committee for the Humanities, Fine Arts and Social 
Sciences and subsequently by the Division of Social Sciences Faculty council. 
The College's approval process also requires that proposals are first vetted in the 
on-line Arts & Science College Challenge, along with consultation with cognate 
departments. 
The proposed Minor is based on existing courses in the Department of 
Sociology; thus there are no resource implications for the College of Arts & 
Science. 
The College believes that the availability of new programs such as the Minor in 
Crime, Law, and Justice Studies, which allows our students to explore a 
discipline outside of their major, has become essential. Students also feel that 
the additional credential provides them with an advantage in their pursuit of a 
career. 
        """
    ]
    
    # Initialize and build system
    system = GraphRAGSystem()
    system.build_knowledge_base(documents)
    
    # Test queries
    test_queries = [
        "In the proposal’s description of program requirements, how are Sociology majors treated differently from non-Sociology majors with respect to credit unit expectations for completing the minor in Crime, Law, and Justice Studies, and what specific restrictions are noted about the double-counting of courses between a student’s major and minor, including the maximum number of credit units permitted in a single subject for the B.A. Four-year degree? ",
        "What prompted the students to express the minor's demand ? (Direct approach in rationale ) OR What factors, according to the proposal, prompted students to express demand for the minor in Crime, Law, and Justice Studies, and how did this demand influence the department’s decision to formally introduce the program?"
    ]
    
    print("=== GRAPH RAG QUERY RESULTS ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = system.query(query)
        print(f"Answer: {result['answer']}")
        print(f"Sources used: {result['retrieval_info']['chunks_found']} chunks, {result['retrieval_info']['entities_found']} entities")
        
        # Test multi-hop reasoning
        reasoning_result = system.multi_hop_reasoning(query)
        if reasoning_result['path_count'] > 0:
            print(f"Reasoning paths found: {reasoning_result['path_count']}")
    
if __name__ == "__main__":
    main()