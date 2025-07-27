import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "wouter";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import kennethImg  from "../images/kenneth.png";
import rajImg      from "../images/raj.png";
import thomasImg   from "../images/thomas.png";
import victoriaImg from "../images/victoria.png";

export default function Team() {
  const people = [
    { 
      id: 1, 
      name: "Kenneth Hahn",  
      role: "Data Scientist",
      email:"hahnkenneth@berkeley.edu",
      description: `Data scientist with four years of manufaturing engineering experience, driving efficiency improvements for Tesla and P&G. 
      Current MIDS student at UC Berkeley with an undergraduate degree in Chemical Engineering from UC Berkeley.`,
      image: kennethImg 
    },
    { 
      id: 2, 
      name: "Raj Jagannath", 
      role: "Backend Engineer", 
      email:"rjagan@berkeley.edu",
      description: `Data scientist with 3 years of software engineering experience at HPE Aruba Networking working on test automation and DevOps.
      Current MIDS student at UC Berkeley with an BS in Computer Science and minor in Managerial Economics from UC Davis.`,
      image: rajImg 
    },
    { 
      id: 3, 
      name: "Thomas Lee",    
      role: "Frontend Engineer", 
      email:"thomascl@berkeley.edu",
      description: "Aspiring data scientist with two years of experience in machine learning engineering. Employed as a data science intern at the East Bay Municipal Utility District supporting the finance department with water projections and pipeline failure prediction. Current MIDS student at UC Berkeley with an undergraduate degree in Computer Science and minor in Data Science.",
      image: thomasImg 
    },
    { 
      id: 4, 
      name: "Victoria Brendel", 
      role: "Machine Learning Engineer", 
      email: "victoriabrendel@berkeley.edu",
      description: `Data scientist with 2 years of experience in manufacturing operations. Employed as an operations engineer at SpaceX
      supporting Falcon and Dragon production. Current MIDS student at UC Berkeley with a BA in Astrophysics from UC Berkeley.`,
      image: victoriaImg 
    }
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-light text-slate-900 mb-3">The Team</h2>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto">Learn more about the team behind the project</p>
      </div>

      {/* Back to Home */}
      <Link href="/">
        <Button variant="outline" className="mb-6">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Home
        </Button>
      </Link>

      {/* Card */}
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="text-2xl font-light">Meet the Core Team</CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          {people.map((person) => (
            <div key={person.id} className="flex items-start space-x-4">
              <img
                src={person.image}
                alt={person.name}
                width={72}
                height={72}
                className="rounded-full object-cover"
              />

              {/* Text block */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between w-full">
                {/* Name + role (left side) */}
                <div>
                  <h4 className="text-lg font-semibold text-slate-900">{person.name}</h4>
                  <p className="text-sm text-primary font-medium">{person.role}</p>
                  <p className="text-sm text-primary text-slate-600">{person.email}</p>
                </div>

                {/* Description (right side on ≥ 640 px; below on mobile) */}
                <p className="text-slate-700 mt-2 sm:mt-0 sm:ml-4">
                  {person.description}
                </p>
              </div>

            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
