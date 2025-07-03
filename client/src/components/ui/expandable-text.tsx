import { useState } from "react";
import { Button } from "@/components/ui/button";
import { truncateText } from "@/lib/utils";

interface ExpandableTextProps {
  text: string;
  maxLength?: number;
  className?: string;
}

export function ExpandableText({ text, maxLength = 150, className = "" }: ExpandableTextProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const needsTruncation = text.length > maxLength;
  
  if (!needsTruncation) {
    return <div className={className}>{text}</div>;
  }

  return (
    <div className={`space-y-2 ${className}`}>
      <div>
        {isExpanded ? text : truncateText(text, maxLength)}
      </div>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsExpanded(!isExpanded)}
        className="text-primary hover:text-primary/80 p-0 h-auto font-medium text-sm"
      >
        {isExpanded ? 'Show less' : 'Show more'}
      </Button>
    </div>
  );
}
