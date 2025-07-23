import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Link } from "wouter";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import Image from "next/image";

export default function Team() {
  const people = [
    {
      id: 1,
      name: "Kenneth Hahn",
      role: "Data Scientist",
      description: "Data Scientist",
      image: "/images/kenneth.png",
    },
    {
      id: 2,
      name: "Raj Jagannath",
      role: "Data Scientist",
      description: "Data Scientist",
      image: "/images/team/raj.png",
    },
    {
      id: 3,
      name: "Thomas Lee",
      role: "Data Scientist",
      description: "Data Scientist",
      image: "/images/team/thomas.png",
    },
    {
      id: 4,
      name: "Victoria Brendel",
      role: "Data Scientist",
      description: "Data Scientist",
      image: "/images/team/victoria.png",
    },
  ];
  
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-light text-slate-900 mb-3">The Team</h2>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto">
          Meet the Team
        </p>
      </div>

      {/* Back to Home Button */}
      <div>
        <Link href="/">
          <Button variant="outline" className="mb-6">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Home
          </Button>
        </Link>
      </div>
      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="text-2xl font-light">Meet the Core Team</CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          {people.map((person) => (
            <div
              key={person.id}
              className="flex items-start space-x-4"
            >
              {/* Avatar */}
              <div className="flex-shrink-0">
                <Image
                  src={person.image}
                  alt={person.name}
                  width={72}
                  height={72}
                  className="rounded-full object-cover"
                />
              </div>

              {/* Text */}
              <div className="space-y-1">
                <h4 className="text-lg font-semibold text-slate-900">
                  {person.name}
                </h4>
                <p className="text-sm text-primary font-medium">
                  {person.role}
                </p>
                <p className="text-slate-700 leading-relaxed">
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
